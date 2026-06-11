srun python main_finetune.py --batch_size 32 --blr 6e-4 --warmup_epochs 10 --epochs 200 --model metaspikformer_8_512 --data_path flowers102_imagenet_like/ --output_dir outputs/55M --log_dir outputs/55M --model_mode ms --dist_eval --num_workers 8 --nb_classes 102

srun python -m torch.distributed.run --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 main_finetune.py --batch_size 32 --blr 6e-4 --warmup_epochs 10 --epochs 200 --model metaspikformer_8_512 --data_path flowers102_imagenet_like/ --output_dir outputs/55M --log_dir outputs/55M --model_mode ms --dist_eval --num_workers 8 --nb_classes 102

# ajuster nproc_per_node en fonction de ce qu'on a de dispo


export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

ssh login@front.convergence.lip6.fr
salloc --nodes=1 --gpus-per-node=a100_3g.40gb:1 --time=60
source venv/bin/activate
module load cuda/11.1 python/anaconda3