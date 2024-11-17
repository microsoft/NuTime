# Pretrain
python main.py --config_path configs/default_ssl.json \
    --dataset_dir $DATA_DIR --archives all-merged --force

# Finetune from ckpt
python main.py --config_path configs/default_finetune.json \
    --load_checkpoint_path ckpt/checkpoint_bias9.pth \
    --dataset_dir $DATA_DIR --archives ucr --force

# Run few shot from ckpt
python main.py --config_path configs/default_finetune.json \
    --load_checkpoint_path ckpt/checkpoint_bias9.pth \ 
    --dataset_dir $DATA_DIR --archives ucr --few_shot --force

# Run from scratch
python main.py --config_path configs/default_cls.json \
    --dataset_dir $DATA_DIR --archives ucr --force

# # Run few shot from scratch
python main.py --config_path configs/default_cls.json \
    --dataset_dir $DATA_DIR --archives ucr --few_shot --force
