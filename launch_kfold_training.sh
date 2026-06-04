#!/usr/bin/env bash
#
# Launch all 8 k-fold training tasks in separate tmux sessions with limited concurrency.
#
#   tasks = {Brightpn, Brightmos} x {2class, 4class} x {supervised, semisup}
#
# Each task:
#   1. trains 5-fold CV  (main_kfold.py / main_unlabelled_kfold.py, --use_indices)
#      -> checkpoints + cv_summary_*.json in   checkpoints_kfold/<task>/
#   2. visualizes the best fold's checkpoint    -> visualizations_kfold/<...>/
#   3. logs everything to                        logs_kfold/<task>.log
#
# Concurrency is capped at MAX_PARALLEL simultaneous tmux sessions (single GPU friendly).
#
# Config via env vars (all optional):
#   MAX_PARALLEL=2  N_FOLDS=5  SUP_EPOCHS=100  SEMISUP_EPOCHS=200
#   ENV_NAME=spectra  CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh  VIZ_METHOD=umap
#
# Usage:
#   ./launch_kfold_training.sh                 # all 8 tasks
#   MAX_PARALLEL=1 ./launch_kfold_training.sh  # one at a time
#   SUP_EPOCHS=2 SEMISUP_EPOCHS=2 ./launch_kfold_training.sh   # quick smoke test
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-spectra}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
N_FOLDS="${N_FOLDS:-5}"
SUP_EPOCHS="${SUP_EPOCHS:-100}"
SEMISUP_EPOCHS="${SEMISUP_EPOCHS:-200}"
VIZ_METHOD="${VIZ_METHOD:-umap}"

CKPT_ROOT="checkpoints_kfold"
VIZ_ROOT="visualizations_kfold"
LOG_DIR="logs_kfold"
RUNNER_DIR="${LOG_DIR}/runners"
SESSION_PREFIX="kfold"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$RUNNER_DIR" "$CKPT_ROOT" "$VIZ_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed." >&2
    exit 1
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda profile not found at $CONDA_SH (set CONDA_SH=...)." >&2
    exit 1
fi

DATASETS=(Brightpn Brightmos)
CLASSES=(2 4)
MODES=(supervised semisup)

running_count() { tmux ls 2>/dev/null | grep -c "^${SESSION_PREFIX}_" || true; }

wait_for_slot() {
    while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
        echo "[launcher] $(running_count)/$MAX_PARALLEL sessions busy — waiting for a free slot..."
        sleep 15
    done
}

launch_task() {
    local data="$1" cls="$2" mode="$3"
    local task="${data}_${cls}cls_${mode}"
    local session="${SESSION_PREFIX}_${task}"
    local ckpt_dir="${CKPT_ROOT}/${task}"
    local log="${LOG_DIR}/${task}.log"
    local runner="${RUNNER_DIR}/run_${task}.sh"

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[launcher] session $session already exists — skipping."
        return
    fi

    local train_cmd summary
    if [ "$mode" = "supervised" ]; then
        train_cmd="python main_kfold.py --use_indices --data ${data} --classes ${cls} --n_folds ${N_FOLDS} --epochs ${SUP_EPOCHS} --out_dir ${ckpt_dir}"
        summary="${ckpt_dir}/cv_summary_${data}_${cls}cls_${N_FOLDS}fold.json"
    else
        train_cmd="python main_unlabelled_kfold.py --use_indices --data ${data} --classes ${cls} --n_folds ${N_FOLDS} --epochs ${SEMISUP_EPOCHS} --out_dir ${ckpt_dir}"
        summary="${ckpt_dir}/cv_summary_semisup_${data}_${cls}cls_${N_FOLDS}fold.json"
    fi

    # Write a self-contained runner script for this task
    cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

echo "[\$(date)] ===== TRAIN START: ${task} ====="
${train_cmd}

echo "[\$(date)] ===== VISUALIZE: ${task} ====="
BEST=\$(python -c "import json; print(json.load(open('${summary}'))['best_checkpoint'])")
echo "Best checkpoint: \$BEST"
python visualize_latent_space.py --checkpoint "\$BEST" --use_indices --out_root "${VIZ_ROOT}" --method "${VIZ_METHOD}"

echo "[\$(date)] ===== DONE: ${task} ====="
EOF
    chmod +x "$runner"

    wait_for_slot
    echo "[launcher] launching $session  (epochs: $([ "$mode" = supervised ] && echo "$SUP_EPOCHS" || echo "$SEMISUP_EPOCHS"), log: $log)"
    tmux new-session -d -s "$session" "bash '$runner' 2>&1 | tee '$log'"
    sleep 2   # let the session register before the next slot check
}

echo "=================================================================="
echo " K-fold training launcher"
echo "   MAX_PARALLEL=$MAX_PARALLEL  N_FOLDS=$N_FOLDS"
echo "   SUP_EPOCHS=$SUP_EPOCHS  SEMISUP_EPOCHS=$SEMISUP_EPOCHS"
echo "   checkpoints -> $CKPT_ROOT/<task>/   visualizations -> $VIZ_ROOT/"
echo "=================================================================="

for data in "${DATASETS[@]}"; do
    for cls in "${CLASSES[@]}"; do
        for mode in "${MODES[@]}"; do
            launch_task "$data" "$cls" "$mode"
        done
    done
done

echo
echo "[launcher] all 8 tasks dispatched (respecting MAX_PARALLEL=$MAX_PARALLEL)."
echo "  Watch sessions:   tmux ls"
echo "  Attach to one:    tmux attach -t ${SESSION_PREFIX}_Brightpn_2cls_supervised"
echo "  Tail a log:       tail -f ${LOG_DIR}/Brightpn_2cls_supervised.log"
echo "  CV summaries:     ${CKPT_ROOT}/<task>/cv_summary_*.json"
