# -*- coding: utf-8 -*-
"""adult_modified.py —— 健壮版 Adult 数据预处理

相对 adult.py 的改动：
1. 不再依赖 `dtype == 'object'` 来判断"要不要做字典映射"。
   pandas 2.x 起字符串列可能是 `string[pyarrow]`，pandas 3.x 起默认就是 `str` dtype，
   原来的 `== 'object'` 判断会全部失效 -> 映射被整体跳过 -> 原始字符串直接进 float32 转换。
   现在改为"先尝试数值化，再对残余字符串做字典映射"的统一流程。
2. CSV 读取改逐行过滤：自动跳过表头行、纯文本注释行、结尾空行、列数不符的残缺行，
   避免 Pandas 把注释/空行当数据行读入。
3. 所有列最终统一 `to_numeric(errors='coerce')` + `fillna(-1)` 兜底，
   保证输出 DataFrame 全为数字，脏数据不会残留成字符串。
4. 标签列已经是 0/1 数字时不会被字符串字典二次映射成 NaN（避免污染整列）。
5. 划分训练/验证/测试前剔除标签不属于 {0,1} 的脏行；
   train_test_split 增加空集 / 单样本保护，避免空集划分报错。
"""

import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# 列定义
# --------------------------------------------------------------------------- #
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "money",
]

LABEL_COLUMNS = ("money", "money1")

# 类别特征映射表（键统一按 strip + lower 后匹配，大小写/首尾空格都不敏感）
CATEGORICAL_MAPS = {
    "workclass": {
        "State-gov": 0, "Self-emp-not-inc": 1, "Private": 2, "Federal-gov": 3,
        "Local-gov": 4, "Self-emp-inc": 5, "Without-pay": 6, "Never-worked": 7,
    },
    "marital-status": {
        "Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2,
        "Married-spouse-absent": 3, "Separated": 4,
        "Married-AF-spouse": 5, "Widowed": 6,
    },
    "occupation": {
        "Adm-clerical": 0, "Exec-managerial": 1, "Handlers-cleaners": 2,
        "Prof-specialty": 3, "Other-service": 4, "Sales": 5,
        "Craft-repair": 6, "Transport-moving": 7, "Farming-fishing": 8,
        "Machine-op-inspct": 9, "Tech-support": 10, "Protective-serv": 11,
        "Armed-Forces": 12, "Priv-house-serv": 13,
    },
    "relationship": {
        "Not-in-family": 0, "Husband": 1, "Wife": 2,
        "Own-child": 3, "Unmarried": 4, "Other-relative": 5,
    },
    "race": {
        "White": 0, "Black": 1, "Asian-Pac-Islander": 2,
        "Amer-Indian-Eskimo": 3, "Other": 4,
    },
    "sex": {"Male": 0, "Female": 1},
    "native-country": {
        "United-States": 0, "Cuba": 1, "Jamaica": 2, "India": 3,
        "Mexico": 4, "South": 5, "Puerto-Rico": 6, "Honduras": 7,
        "England": 8, "Canada": 9, "Germany": 10, "Iran": 11,
        "Philippines": 12, "Italy": 13, "Poland": 14, "Columbia": 15,
        "Cambodia": 16, "Thailand": 17, "Ecuador": 18, "Laos": 19,
        "Taiwan": 20, "Haiti": 21, "Portugal": 22,
        "Dominican-Republic": 23, "El-Salvador": 24, "France": 25,
        "Guatemala": 26, "China": 27, "Japan": 28, "Yugoslavia": 29,
        "Peru": 30, "Outlying-US(Guam-USVI-etc)": 31, "Scotland": 32,
        "Trinadad&Tobago": 33, "Greece": 34, "Nicaragua": 35,
        "Vietnam": 36, "Hong": 37, "Ireland": 38, "Hungary": 39,
        "Holand-Netherlands": 40,
    },
}

# 标签映射：同时兼容 "<=50K" / "<=50K." 两种写法（train 无点、test 有点）
LABEL_MAP = {
    "<=50k": 0, "<=50k.": 0, "0": 0, "0.0": 0,
    ">50k": 1, ">50k.": 1, "1": 1, "1.0": 1,
}

# 众数填充的列（Adult 标准缺失位置）
MODE_FILL_COLUMNS = ("workclass", "occupation", "native-country")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _is_text_col(series):
    """判断一列是否为字符串/文本类型，跨 pandas 版本安全。

    pandas < 2 : dtype == object
    pandas 2.x : 可能是 string[python] / string[pyarrow]
    pandas 3.x : 默认 str (StringDtype)
    """
    dtype = series.dtype
    if dtype == object:
        return True
    try:
        if isinstance(dtype, pd.StringDtype):
            return True
    except Exception:
        pass
    return str(dtype) in ("str", "string", "string[python]", "string[pyarrow]")


def _looks_number(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _is_comment_line(line):
    stripped = line.lstrip()
    if not stripped:
        return True
    return stripped.startswith(("#", "//", "--", "%", "|", "*"))


def _read_clean_csv(path, columns=COLUMNS):
    """逐行读取 CSV，过滤掉表头 / 注释 / 空行 / 列数不符的残缺行。

    相比 `pd.read_csv(...)` 直接读，这里能保证：
      - 文件末尾的纯文本注释、空行不会被当成数据；
      - 字段首尾空格被清除；
      - 列数不对的坏行被丢弃而不是抛 ParserError；
      - 表头行（首字段非数字）被识别并跳过。
    """
    n_cols = len(columns)
    rows = []

    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        rows = []
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if _is_comment_line(line):
                        continue
                    try:
                        fields = next(csv.reader([line]))
                    except Exception:
                        continue
                    if len(fields) != n_cols:
                        continue  # 列数不符：注释、残缺行、多余分隔符
                    fields = [f.strip() for f in fields]
                    if not _looks_number(fields[0]):
                        continue  # 表头行 / 纯文本说明行
                    rows.append(fields)
        except UnicodeDecodeError:
            continue
        break

    if not rows:
        raise ValueError("文件 %s 中没有解析出任何有效数据行" % path)

    return pd.DataFrame(rows, columns=list(columns))


def _to_numeric(series, mapping=None, fill_value=-1.0):
    """把一列安全地转换成数值。

    顺序：
      1. 先尝试 `pd.to_numeric`，若整列都能转成数字，直接返回（不再做字典映射，
         避免"已经是数字的标签被字符串字典映射成 NaN"这类类型污染问题）；
      2. 否则把残余的字符串 strip + lower 后按 mapping 映射，再数值化；
      3. 仍然无法识别的（含 NaN、未知类别、'?'）统一填 `fill_value`（默认 -1）。
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(np.float64)

    if mapping:
        normalised = {str(k).strip().lower(): v for k, v in mapping.items()}
        keys = series.astype("string").str.strip().str.lower()
        mapped = pd.to_numeric(keys.map(normalised), errors="coerce")
        numeric = numeric.combine_first(mapped)

    return numeric.fillna(fill_value).astype(np.float64)


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load_adult_dataset(data_dir):
    """读取并返回 (df_train, df_test)，标签列分别为 money / money1。"""
    candidates_train = [
        os.path.join(data_dir, "adult", "train.csv"),
        os.path.join(data_dir, "train.csv"),
    ]
    candidates_test = [
        os.path.join(data_dir, "adult", "test.csv"),
        os.path.join(data_dir, "test.csv"),
    ]

    samples_path = next((p for p in candidates_train if os.path.exists(p)), None)
    targets_path = next((p for p in candidates_test if os.path.exists(p)), None)
    if samples_path is None or targets_path is None:
        raise FileNotFoundError(
            "在 %s 下找不到 adult/train.csv 或 adult/test.csv" % data_dir
        )

    df_train = _read_clean_csv(samples_path)
    df_test = _read_clean_csv(targets_path)
    # test 的标签列统一命名为 money1，保持与原实现一致的接口
    df_test = df_test.rename(columns={COLUMNS[-1]: "money1"})
    return df_train, df_test


# --------------------------------------------------------------------------- #
# 清洗
# --------------------------------------------------------------------------- #
def dataCleaning(dataSet):
    """把任意脏的 Adult DataFrame 清洗成"全数字、无 NaN"的 DataFrame。"""
    df = dataSet.copy()

    # 1. 删除不相关的列（education 与 education_num 重复，保留数值版）
    if "education" in df.columns:
        df = df.drop(columns=["education"])

    # 2. 文本列：去首尾空格，空串视为缺失，'?' 视为缺失
    for col in df.columns:
        if _is_text_col(df[col]):
            stripped = df[col].astype("string").str.strip()
            stripped = stripped.replace("?", np.nan)
            stripped = stripped.replace("", np.nan)
            df[col] = stripped

    # 3. 已知缺失列用众数填充
    for col in MODE_FILL_COLUMNS:
        if col in df.columns and _is_text_col(df[col]):
            mode = df[col].mode(dropna=True)
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])

    # 4. 类别列映射（值感知：已经是数字的列不会被再映射一次）
    for col, mapping in CATEGORICAL_MAPS.items():
        if col in df.columns:
            df[col] = _to_numeric(df[col], mapping)

    # 5. 标签列映射（兼容 "<=50K" / "<=50K." 两种写法）
    for col in LABEL_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric(df[col], LABEL_MAP)

    # 6. 兜底：所有剩余列统一数值化，无法识别的填 -1，确保能转 float32
    for col in df.columns:
        if _is_text_col(df[col]):
            df[col] = _to_numeric(df[col], CATEGORICAL_MAPS.get(col))
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(-1.0).astype(np.float64)
    return df


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class AdultsDataset(Dataset):
    def __init__(self, data, is_train=True, mean=None, std=None):
        self.num_classes = 2
        self.is_train = is_train

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        label_col = next((c for c in LABEL_COLUMNS if c in data.columns), None)
        if label_col is None:
            raise ValueError("标签列不存在！")

        feature_cols = [c for c in data.columns if c != label_col]
        features = data[feature_cols].apply(pd.to_numeric, errors="coerce")
        labels = pd.to_numeric(data[label_col], errors="coerce")

        self.data = np.nan_to_num(
            features.to_numpy(dtype=np.float64), nan=-1.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        self.labels = np.nan_to_num(
            labels.to_numpy(dtype=np.float64), nan=-1.0
        ).astype(np.int64)

        # 计算均值和标准差（仅训练集）
        if self.is_train:
            if mean is None or std is None:
                self.mean = np.mean(self.data, axis=0)
                self.std = np.std(self.data, axis=0)
            else:
                self.mean = np.asarray(mean, dtype=np.float32)
                self.std = np.asarray(std, dtype=np.float32)
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            if mean is not None and std is not None:
                self.mean = np.asarray(mean, dtype=np.float32)
                self.std = np.asarray(std, dtype=np.float32)
                self.data = (self.data - self.mean) / (self.std + 1e-8)
            else:
                self.mean = None
                self.std = None

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        if isinstance(index, (list, np.ndarray, torch.Tensor)):
            if len(index) > 0:
                index = index[0]
            else:
                raise IndexError("Received empty index list in __getitem__.")
        index = int(index)
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def _safe_split(df, test_size=0.2, random_state=42):
    """在小样本 / 空集情况下也不会崩溃的 train_test_split。"""
    n = len(df)
    if n == 0:
        empty = df.iloc[0:0].copy()
        return empty, empty
    if n == 1:
        return df.copy(), df.iloc[0:0].copy()
    n_test = int(round(n * test_size))
    n_test = min(max(n_test, 1), n - 1)
    return train_test_split(df, test_size=n_test, random_state=random_state)


def _filter_valid_labels(df, label_col, verbose=True):
    """剔除标签不属于 {0,1} 的脏行，避免后续空集划分。"""
    if label_col not in df.columns:
        raise ValueError("缺少标签列 %s" % label_col)
    mask = df[label_col].isin([0.0, 1.0])
    dropped = int((~mask).sum())
    if dropped and verbose:
        print("[adult_modified] %s 列剔除 %d 行非法标签（值不在 {0,1}）"
              % (label_col, dropped))
    return df[mask].reset_index(drop=True)


def adult(data_path, balance=True):
    """与原 adult.py 完全一致的返回签名，可直接替换调用。"""
    df_train, df_test = load_adult_dataset(data_dir=data_path)
    df_train = dataCleaning(df_train)
    df_test = dataCleaning(df_test)

    # 剔除标签异常的脏行（例如无法映射的标签被兜底成 -1）
    df_train = _filter_valid_labels(df_train, "money")
    # 若 test 没被命名为 money1（个别数据只有一个文件），退化为 money
    test_label = "money1" if "money1" in df_test.columns else "money"
    df_test = _filter_valid_labels(df_test, test_label)

    if len(df_train) == 0:
        raise ValueError("训练集清洗后为空，请检查数据文件与标签列。")

    if balance:
        # 按标签各自 80/20 划分为训练集和验证集，保持正负样本比例
        data_label_0 = df_train[df_train["money"] == 0]
        data_label_1 = df_train[df_train["money"] == 1]
        df_label_0_train, df_label_0_val = _safe_split(data_label_0, 0.2)
        df_label_1_train, df_label_1_val = _safe_split(data_label_1, 0.2)
        df_train = pd.concat([df_label_0_train, df_label_1_train]).reset_index(drop=True)
        df_val = pd.concat([df_label_0_val, df_label_1_val]).reset_index(drop=True)
    else:
        data_label_0 = df_train[df_train["money"] == 0]
        data_label_1 = df_train[df_train["money"] == 1]

        df_label_0_train, df_label_0_val = _safe_split(data_label_0, 0.2)
        df_label_1_train, df_label_1_val = _safe_split(data_label_1, 0.2)

        # 训练集：全部标签为 1 的样本 + 80% 标签为 0 的样本
        df_train = pd.concat([df_label_1_train, df_label_0_train]).reset_index(drop=True)

        # 验证集：只保留标签为 0 的剩余部分（与原实现语义一致）
        if len(df_label_1_val) >= 2:
            df_label_1_val_1, df_label_1_val_2 = _safe_split(df_label_1_val, 0.5)
        df_val = df_label_0_val

        # 测试集：只保留标签为 0 的样本（与原实现语义一致）
        df_test = df_test[df_test[test_label] == 0]

    if len(df_val) == 0:
        # 极端情况下验证集为空，用测试集回退，避免下游报错
        df_val = df_test.copy()

    # 定义类别名称
    class_names = ["<=50K", ">50K"]

    # 创建训练数据集，计算均值和标准差
    train_dataset = AdultsDataset(df_train, is_train=True)
    mean = train_dataset.mean
    std = train_dataset.std

    # 创建验证和测试数据集，使用训练集的均值和标准差
    val_dataset = AdultsDataset(df_val, is_train=False, mean=mean, std=std)
    test_dataset = AdultsDataset(df_test, is_train=False, mean=mean, std=std)

    channel = 1
    im_size = train_dataset.data.shape[1]
    num_classes = train_dataset.num_classes

    return (channel, im_size, num_classes, class_names, mean, std,
            train_dataset, val_dataset, test_dataset)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _self_test(data_path):
    channel, im_size, num_classes, class_names, mean, std, tr, va, te = adult(data_path)
    print("channel      :", channel)
    print("im_size      :", im_size)
    print("num_classes  :", num_classes)
    print("class_names  :", class_names)
    print("train/val/test:", len(tr), len(va), len(te))
    print("train data dtype:", tr.data.dtype, "shape", tr.data.shape)
    print("train label dtype:", tr.labels.dtype, "unique", np.unique(tr.labels))
    print("test  label unique:", np.unique(te.labels))
    print("NaN in train data:", bool(np.isnan(tr.data).any()))
    print("sentinel(-1) count per feature:", (tr.data == -1).sum(axis=0)[:5], "...")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/EVA/data"
    _self_test(path)
