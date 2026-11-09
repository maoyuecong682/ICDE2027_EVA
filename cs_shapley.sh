#!/bin/bash

# 定义参数
methods=("cs_shapley")
# selection_fractions=(0.01 0.1 0.3 0.5 0.7 0.9)
selection_fractions=(0.9)
# datasets=("adult" "imdb" "cifar100" "news" "covtype" "HR" "number" "mnist2mnistm" "tinyimagenet")
datasets=("tinyimagenet")

# 定义日志文件
log_file="./error_log/cs_shapley.log"
mkdir -p "$(dirname "$log_file")"  # 确保日志目录存在
# > "$log_file"  # 清空日志文件

# 根据 dataset 映射 data_path 和 model
get_data_path_and_model() {
    case "$1" in
        "news")
            echo "/data2/wyh-dataset/text-dataset/News TextMLP"
            ;;
        "imdb")
            echo "/data2/wyh-dataset/text-dataset/imdb TextMLP"
            ;;
        "cifar10"|"cifar100"|"mnist2mnistm"|"number"|"tinyimagenet")
            echo "/data2/wyh-dataset/image-dataset ResNet18"
            ;;
        "adult"|"covtype"|"HR"|"IMDBLarge")
            echo "/data2/wyh-dataset/tabular-dataset MLP"
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
            # 执行命令
            echo "Running: method=$method, dataset=$dataset, data_path=$data_path, model=$model, selection_fraction=$selection_fraction"
            
            # 捕获错误输出并记录
            start_time=$(date +"%Y-%m-%d %H:%M:%S")
            {
                python main1.py \
                    --data_path "$data_path" \
                    --dataset "$dataset" \
                    --method "$method" \
                    --epochs 100 \
                    --model "$model" \
                    --selection_fraction "$selection_fraction" \
                    --selection_lr 0.1 \
                    --balance 0 \
                    --patience 100 
            } || {
                end_time=$(date +"%Y-%m-%d %H:%M:%S")
                echo "[ERROR] [$start_time] Method: $method, Dataset: $dataset, Selection Fraction: $selection_fraction" >> "$log_file"
                echo "Details: An error occurred. Check the output logs for more information." >> "$log_file"
                echo "[ERROR END] [$end_time]" >> "$log_file"
                echo >> "$log_file"
            }
        done
    done
done

# 提示完成
echo "All tasks completed. Check $log_file for errors."