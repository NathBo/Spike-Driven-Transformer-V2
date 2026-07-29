#!/bin/bash
#SBATCH --job-name=imagenet_finetune
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Vu que tu as mis --num_workers 8
#SBATCH --gres=gpu:a100_3g.40gb:1            # C'EST CETTE LIGNE QUI MANQUAIT !
#SBATCH --time=150:00:00         # Ajuste le temps selon tes besoins
source ../venv/bin/activate
module load cuda/11.1 python/anaconda3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
srun python -m torch.distributed.run --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 main_finetune.py --batch_size 32 --blr 6e-4 --warmup_epochs 0 --epochs 200 --model metaspikformer_8_512 --data_path imagenet_kaggle/ --output_dir outputs/55Mimagenet --log_dir outputs/55Mimagenet --model_mode ms --dist_eval --num_workers 8 --nb_classes 1000
