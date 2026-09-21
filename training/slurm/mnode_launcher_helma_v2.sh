#!/bin/bash -l
#
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --partition=h200
#SBATCH --nodes=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

unset SLURM_EXPORT_ENV

export http_proxy="http://proxy.example.edu:80"
export https_proxy="http://proxy.example.edu:80"
export no_proxy="localhost,127.0.0.1,10.0.0.1,10.0.0.2"
export NO_PROXY="$no_proxy"

cd /home/hpc/$GROUP/$USER/project/CTFlow-v2
mkdir -p logs

export SIF_IMAGE="/path/to/workspace/tmi_container_v2.sif"
echo "Singularity set to: $SIF_IMAGE"

export NCCL_DEBUG=info
export NCCL_PROTO=simple
export NCCL_SHARP_ENABLE=1
export NCCL_IB_HCA=mlx5_0

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_LOG_LEVEL=1
export TORCH_LOGS="-dynamo"

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(echo $HOSTNAMES | awk '{print $1}').your-cluster-domain.edu
export MASTER_PORT=12802
export COUNT_NODE=$(echo $HOSTNAMES | wc -w)
export NUM_GPU=$(nvidia-smi -L | wc -l)

echo "Launching on $COUNT_NODE nodes, total $NUM_GPU GPUs per node"
echo "Node list: $HOSTNAMES"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

export SCRIPT="ctflow/lvfm/train_ft.py"
export CONFIG="ctflow/lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all_ft.yaml"

for file in "$SCRIPT" "$CONFIG"; do
  if [ ! -f "$file" ]; then
    echo "ERROR: $file not found. Abort."
    exit 1
  else
    echo "Found $file"
  fi
done

srun slurms/trainer_helma_ft_v2.sh
