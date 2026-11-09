python main0.py --data_path /data2/wyh-dataset/image-dataset --dataset number --method full --epochs 100 --model ResNet18 --selection_fraction 0.2 --selection_lr 0.1 --balance_target 0 --gpu +1 --selection_interval 20 --selection_framework internal --patience 100 --uncertainty_type BvSB

python main0.py --data_path /data2/wyh-dataset/image-dataset --dataset cifar10 --method herding --epochs 100 --model ViT --selection_fraction 0.1 --selection_lr 0.1 --balance_target 0 --selection_interval 20 --patience 3 --selection_framework active --uncertainty_type BvSB

python main0.py --data_path /data2/wyh-dataset/tabular-dataset --dataset covtype --method kmeans --epochs 100 --model TabNet --selection_fraction 0.1 --selection_lr 0.001 --balance_target 1 --selection_interval 20 --patience 100 --task_type classification

python main0.py --data_path /data2/wyh-dataset/tabular-dataset --dataset House --method kmeans --epochs 100 --model REMLP --selection_fraction 0.1 --selection_lr 0.001 --balance_target 0 --selection_interval 20 --patience 100 --task_type regression