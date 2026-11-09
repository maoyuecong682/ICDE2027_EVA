#!/bin/bash

# 定义参数
methods=("cg_influence" "Arnoldi" "LiSSA" "TracIn")
selection_fractions=(0.01 0.1 0.3 0.5 0.7 0.9)
balance_targets=(0 1)
# selection_framework=("once" "interval")
datasets=("adult" "cifar10" "imdb" "cifar100" "news" "covtype")

# 根据 dataset 映射 data_path 和 model
get_data_path_and_model() {
    case "$1" in
        "news")
            echo "/data/wyh-dataset/text-dataset/News MLP"
            ;;
        "imdb")
            echo "/data/wyh-dataset/text-dataset/imdb MLP"
            ;;
        "cifar10"|"cifar100")
            echo "/data/wyh-dataset/image-dataset ResNet18"
            ;;
        "adult"|"covtype")
            echo "/data/wyh-dataset/tabular-dataset MLP"
            ;;
        *)
            echo "Unknown dataset"
            exit 1
            ;;
    esac
}

# 遍历所有组合
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        # 获取对应的 data_path 和 model
        result=$(get_data_path_and_model "$dataset")
        data_path=$(echo "$result" | awk '{print $1}')
        model=$(echo "$result" | awk '{print $2}')

        for selection_fraction in "${selection_fractions[@]}"; do
            for balance_target in "${balance_targets[@]}"; do
                # 执行命令
                echo "Running: method=$method, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction, balance_target=$balance_target, selection_framework=once"
                python main.py \
                    --data_path "$data_path" \
                    --dataset "$dataset" \
                    --method "$method" \
                    --epochs 100 \
                    --model "$model" \
                    --selection_fraction "$selection_fraction" \
                    --selection_lr 0.1 \
                    --balance_target "$balance_target" \
                    --selection_interval 20 \
                    --selection_framework "once" 
            done
        done
    done
done