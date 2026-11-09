#!/bin/bash

# 定义参数
methods=("kcentergreedy")
selection_fractions=(0.01 0.1 0.3 0.5 0.7 0.9)
datasets=("adult" "imdb" "news" "covtype")

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

# 遍历所有组合
for dataset in "${datasets[@]}"; do
    # 获取对应的 data_path 和 model
    result=$(get_data_path_and_model "$dataset")
    data_path=$(echo "$result" | awk '{print $1}')
    model=$(echo "$result" | awk '{print $2}')
    for selection_fraction in "${selection_fractions[@]}"; do
        # 执行命令
        echo "Running: method=kcentergreedy, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction"
        python main0.py \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --method "kcentergreedy" \
            --epochs 100 \
            --model "$model" \
            --selection_fraction "$selection_fraction" \
            --selection_lr 0.1 \
            --patience 100
    done
done
