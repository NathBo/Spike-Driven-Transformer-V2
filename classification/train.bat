python main_finetune.py --batch_size 128 --blr 6e-4 --warmup_epochs 10 --epochs 200 --model metaspikformer_8_512 --data_path flowers102_imagenet_like/ --output_dir outputs/55M --log_dir outputs/55M --model_mode ms --dist_eval

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0