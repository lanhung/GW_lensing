#!/usr/bin/env bash
# SLURM array job: fully simultaneous GW strain + EM lens-model inference, one event per task.
#
#   ls inputs/gwem/*.json > gwem_events.txt
#   sbatch --array=1-$(wc -l < gwem_events.txt)%20 scripts/slurm_gwem.sh gwem_events.txt results/gwem_full
#
# Extra arguments after OUT_DIR are passed to gwlens.joint_gwem full, e.g.
#   --cosmology lcdm --sigma-kext 0.025 --lens-model EPL_shear --assignment marginalise --nlive 500
# The inputs are self-contained JSON files (scripts/prepare_gwem_inputs.sh), so the cluster needs
# only this repository and its Python environment (requirements.txt), not the catalogues.
# Runs resume from bilby/dynesty checkpoints (every 10 min) if a task is requeued.
#SBATCH --job-name=gwem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/gwem_%A_%a.out
set -euo pipefail
LIST=$1; OUT=$2; shift 2
EVENT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=${PYTHONPATH:-.}
srun python -m gwlens.joint_gwem full --event "$EVENT" --out-dir "$OUT" --npool "${SLURM_CPUS_PER_TASK}" "$@"
