#!/usr/bin/env bash

# the following come from main sh.:
#   HOSTNAMES, MASTER_ADDR, MASTER_PORT, COUNT_NODE, NUM_GPU, SLURM_JOB_ID, SIF_IMAGE

export SCRIPT="ctflow/lvfm/train_ft.py"
export CONFIG="ctflow/lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all_ft.yaml"

# rank index
H=$(hostname | cut -d'.' -f1)
THEID=$(echo -e "$HOSTNAMES" | tr ' ' '\n' | nl -w1 -s' ' \
  | awk -v host="$H" '$2==host{print $1-1}')
INDEX=$THEID

MASTER_HOST=$(echo "$MASTER_ADDR" | cut -d'.' -f1)
if [ "$H" = "$MASTER_HOST" ]; then
    echo "This node ($H) is the master node."
else
    echo "This node ($H) is a worker node."
fi

# proxy
export http_proxy="http://proxy.example.edu:80"
export https_proxy="http://proxy.example.edu:80"
export no_proxy="localhost,127.0.0.1,10.0.0.1,10.0.0.2"
export NO_PROXY="$no_proxy"

cd /home/hpc/$GROUP/$USER/project/CTFlow-v2
mkdir -p logs

# ===== paths =====
export SCRATCH_ROOT=/scratch/${USER}/latte_ft_${SLURM_JOB_ID}
export LATENT_TAR_DIR=/path/to/workspace/CT-RATE_latents_ft/CT-RATE_latents_ft

LATENT_TARS=(
  latents_00.tar
  latents_01.tar
  latents_02.tar
  latents_03.tar
  latents_04.tar
  latents_05.tar
  latents_06.tar
  latents_07.tar
  latents_08.tar
  latents_09.tar
  latents_10.tar
  latents_11.tar
  latents_12.tar
  latents_13.tar
  latents_14.tar
  latents_15.tar
)

mkdir -p "$SCRATCH_ROOT"
export SCRATCH_LATENTS=$SCRATCH_ROOT/latents
mkdir -p "$SCRATCH_LATENTS"

# ===== latents: each node extracts TARS_PER_NODE tars starting at INDEX*TARS_PER_NODE =====
TARS_PER_NODE=2
IDX_START=$(( INDEX * TARS_PER_NODE ))
IDX_END=$(( IDX_START + TARS_PER_NODE - 1 ))

for i in $(seq "$IDX_START" "$IDX_END"); do
    if [ "$i" -lt "${#LATENT_TARS[@]}" ]; then
        full_latent="$LATENT_TAR_DIR/${LATENT_TARS[$i]}"
        echo "[Node $INDEX] Extracting ${LATENT_TARS[$i]} to $SCRATCH_LATENTS"
        if [ -f "$full_latent" ]; then
            first_entry=$(tar -tf "$full_latent" | head -n1)
            strip_count=$(printf '%s' "$first_entry" | tr -cd '/' | wc -c)
            tar -xf "$full_latent" -C "$SCRATCH_LATENTS" --strip-components="$strip_count"
            echo "    ✅ Done: ${LATENT_TARS[$i]} (strip-components=$strip_count)"
        else
            echo "    ❌ File not found: $full_latent"
        fi
    else
        echo "[Node $INDEX] No latent tar assigned for index $i. Skipping."
    fi
done

echo "🔢 Counting extracted files..."
echo -n "📁 Latents: "; find "$SCRATCH_LATENTS" -type f -name "*.pt" | wc -l

# no embedding extraction needed — text encoded on-the-fly from CSVs
export LATTE_TRAIN_DATA_ROOT=$SCRATCH_LATENTS
export LATTE_VALID_DATA_ROOT=$SCRATCH_LATENTS

TMP_CONFIG=$SCRATCH_ROOT/config_${INDEX}.yaml
envsubst < $CONFIG > $TMP_CONFIG
CONFIG=$TMP_CONFIG
echo "✅ Generated per-node config: $CONFIG"

container_cmd="singularity exec --nv --pwd $PWD $SIF_IMAGE"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD:$PYTHONPATH
export ACCELERATE_DISABLE_RNG_SYNC=1

echo "Preparing accelerate command for node $INDEX"
if [ "$COUNT_NODE" -eq 1 ]; then
    accelerate_cmd="accelerate launch \
        --num_processes 4 \
        --multi_gpu \
        --num_machines 1 \
        --mixed_precision bf16 \
        $SCRIPT \
        --config $CONFIG"
else
    accelerate_cmd="accelerate launch \
        --num_processes $((4 * COUNT_NODE)) \
        --num_machines $COUNT_NODE \
        --multi_gpu \
        --num_cpu_threads_per_process 32 \
        --mixed_precision bf16 \
        --machine_rank $INDEX \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        $SCRIPT \
        --config $CONFIG"
fi

export MPLCONFIGDIR=/tmp/mpl_${SLURM_NODEID:-$INDEX}
mkdir -p "$MPLCONFIGDIR"
$container_cmd python3 -c "import matplotlib; matplotlib.use('Agg')" 2>/dev/null || true

train_cmd="$container_cmd $accelerate_cmd"
echo "Launching training on node $INDEX:"
echo "$train_cmd"
$train_cmd
