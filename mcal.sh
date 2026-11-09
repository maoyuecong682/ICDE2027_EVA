#!/bin/bash

# 定义参数
methods=("mcal")
uncertainty_types=("BvSB" "max_entropy" "max_variation_ratio" "mean_std")
selection_fractions=(0.01 0.1 0.3 0.5 0.7 0.9)
selection_framework=("active_learning")
datasets=("adult" "imdb" "cifar10" "cifar100" "news" "covtype")

# 错误日志文件
error_log="mcal_error_log.txt"

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
for type in "${uncertainty_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        # 获取对应的 data_path 和 model
        result=$(get_data_path_and_model "$dataset")
        data_path=$(echo "$result" | awk '{print $1}')
        model=$(echo "$result" | awk '{print $2}')
        for selection_fraction in "${selection_fractions[@]}"; do
            # 构造命令
            command="python main0.py \
                --data_path \"$data_path\" \
                --dataset \"$dataset\" \
                --method \"mcal\" \
                --epochs 100 \
                --model \"$model\" \
                --selection_fraction \"$selection_fraction\" \
                --selection_lr 0.1 \
                --selection_interval 20 \
                --selection_framework \"active_learning\" \
                --uncertainty_type \"$type\" \
                --patience 100"

            # 打印并运行命令
            echo "Running: method=mcal, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction, selection_framework=active_learning, uncertainty_type=$type"
            eval $command
            # 检查命令是否成功执行
            if [ $? -ne 0 ]; then
                echo "$(date): Error occurred while running: $command" >> "$error_log"
            fi
        done
    done
done