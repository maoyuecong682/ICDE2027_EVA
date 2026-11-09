"""imdb_modified.py -- 修正 imdb.py 里两处写死的路径。

原实现有两处只在原作者机器上成立的绝对路径：

1) BERT 权重（imdb.py 第 247-248 行，两处）
       BertModel.from_pretrained('/data2/wyh-dataset/text-dataset/bert-base-uncase')
   本机没有 /data2，而同目录的 imdb0.py 用的是
       /root/autodl-tmp/bert-base-uncased/models/google-bert--bert-base-uncased/snapshots/master
   同一个 backbone、同一个 tokenizer，换成后者不改变任何数值口径。

2) 嵌入缓存目录（imdb.py 第 112 行）
       embedding_dir = "./embedding"
   相对路径，落到**调用进程的 cwd**上。三个 baseline 各自的 cwd 是各自的仓库根
   （COPS-main / GM-Matching-main / dual-pruning），于是同一份 IMDB 会被缓存三遍，
   而且"有没有命中缓存"取决于从哪儿启动 —— 直接导致跨方法不可比。
   改成数据盘上的绝对路径，与 news.py 用的 data/embedding 是同一个目录。

只改这两处字面量，其余逻辑（打乱、8:2 切分、用训练集均值方差标准化、
pooler_output 768 维）完全沿用原实现。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "imdb.py")

_BERT_NEW = ("/root/autodl-tmp/bert-base-uncased/models/"
             "google-bert--bert-base-uncased/snapshots/master")
_BERT_OLD = "/data2/wyh-dataset/text-dataset/bert-base-uncase"

_EMB_NEW = '/root/autodl-tmp/EVA/data/embedding'
_EMB_OLD = re.compile(r'^[ \t]*embedding_dir\s*=\s*"\./embedding"\s*$', re.M)


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()

    n_bert = src.count(_BERT_OLD)
    if n_bert != 2:
        raise RuntimeError(
            "imdb_modified: 期望 2 处原 BERT 路径，实际 %d 处；请检查 imdb.py" % n_bert
        )
    src = src.replace(_BERT_OLD, _BERT_NEW)

    src, n_emb = _EMB_OLD.subn('    embedding_dir = "%s"' % _EMB_NEW, src)
    if n_emb != 1:
        raise RuntimeError(
            "imdb_modified: 期望恰好 1 处 embedding_dir，实际 %d 处" % n_emb
        )

    if not re.search(r"^import os\b", src, re.M):
        src = "import os\n" + src

    name = "_imdb_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()

imdb = _impl.imdb
ImdbDataset = _impl.ImdbDataset
