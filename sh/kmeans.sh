#!/bin/bash

# 定义参数
methods=("kmeans")
# methods=("IAS" "cs_shapley")
# uncertainty_types=("BvSB" "max_entropy" "max_variation_ratio" "mean_std")
selection_fractions=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
balance_targets=(1)
selection_framework=("interval")
datasets=("cifar10")

# 根据 dataset 映射 data_path 和 model
get_data_path_and_model() {
    case "$1" in
        "news")
            echo "/data2/wyh-dataset/text-dataset/News MLP"
            ;;
        "imdb")
            echo "/data2/wyh-dataset/text-dataset/imdb MLP"
            ;;
        "cifar10"|"cifar100")
            echo "/data2/wyh-dataset/image-dataset ResNet18"
            ;;
        "adult"|"covtype")
            echo "/data2/wyh-dataset/tabular-dataset MLP"
            ;;
        *)
            echo "Unknown dataset"
            exit 1
            ;;
    esac
}

# 创建 error.log 文件
error_log="error.log"
# > "$error_log"  # 清空日志文件

# 遍历所有组合
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        # 获取对应的 data_path 和 model
        result=$(get_data_path_and_model "$dataset")
        data_path=$(echo "$result" | awk '{print $1}')
        model=$(echo "$result" | awk '{print $2}')
        for selection_fraction in "${selection_fractions[@]}"; do
            for balance_target in "${balance_targets[@]}"; do
                for framework in "${selection_framework[@]}"; do
                    # 执行命令
                    echo "Running: method=$method, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction, balance_target=$balance_target, selection_framework=$framework"
                    python main0.py \
                        --data_path "$data_path" \
                        --dataset "$dataset" \
                        --method "$method" \
                        --epochs 100 \
                        --model "$model" \
                        --selection_fraction "$selection_fraction" \
                        --selection_lr 0.1 \
                        --balance_target "$balance_target" \
                        --selection_interval 20 \
                        --selection_framework "$framework"

                    # 检查命令是否成功
                    if [ $? -ne 0 ]; then
                        echo "Error: method=$method, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction, balance_target=$balance_target, selection_framework=$framework" >> "$error_log"
                    fi
                done
            done
        done
    done
done