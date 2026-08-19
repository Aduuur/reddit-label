#!/bin/bash
# =============================================================================
# run_labeling.sh  -  All-in-One Labeling-Job
# -----------------------------------------------------------------------------
# Ein einziger SLURM-Job der:
#   1. Modell + Container + Code aus LSDF in den Workspace holt (falls nötig)
#   2. Input-Daten aus LSDF holt
#   3. Den vLLM-Server INTERN auf diesem Node startet (Hintergrund)
#   4. Die Label-Pipeline gegen localhost laufen lässt
#   5. Ergebnisse nach LSDF zurückschreibt
#
# KEIN SSH-Tunnel nötig - Server und Pipeline laufen auf demselben Node.
#
# Aufruf:
#   sbatch run_labeling.sh INPUTNAME
# Beispiel:
#   sbatch run_labeling.sh conversation_threads_flat.ndjson
#
# INPUTNAME ist der Dateiname in LSDF unter data/ (siehe LSDF_DATA).
# =============================================================================
#SBATCH --job-name=label_llm
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=label_%j.out

set -uo pipefail

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
LSDF_PROJECT=/lsdf/kit/itz/projects/delib_lab
LSDF_DATA=$LSDF_PROJECT/data          # Input-Daten (LSDF)
LSDF_CODE=$LSDF_PROJECT/code          # Pipeline-Code (LSDF)
LSDF_RESULTS=$LSDF_PROJECT/results    # Ergebnisse (LSDF)
MODEL_REL=hf_cache/models/meta-llama/Llama-3.1-70B-Instruct

# Fester Output-Name pro Input (KEIN Zeitstempel) -> ermöglicht nahtlose
# Wiederaufnahme: Ein abgebrochener Lauf wird beim erneuten sbatch fortgesetzt,
# weil die Pipeline bereits gelabelte (comment_id, task)-Paare überspringt.
INPUT_NAME="${1:-conversation_threads_flat.ndjson}"
OUTPUT_NAME="labels_${INPUT_NAME%.*}.ndjson"

PORT=8000
BASE_URL="http://localhost:${PORT}"

echo "============================================================"
echo "LABELING JOB   $(date)"
echo "  Node:       $(hostname)"
echo "  Input:      $INPUT_NAME"
echo "  Output:     $OUTPUT_NAME"
echo "============================================================"

# ----------------------------------------------------------------------------
# 1. Workspace vorbereiten (Modell + Container aus LSDF holen falls nötig)
# ----------------------------------------------------------------------------
# WICHTIG: LSDF ist auf GPU-Nodes NICHT verfügbar. Deshalb müssen Modell und
# Container VORHER im Workspace liegen. Wir prüfen das und brechen sonst ab
# mit klarer Anweisung.
if ! ws_find llm_run >/dev/null 2>&1; then
    echo "FEHLER: Workspace 'llm_run' existiert nicht."
    echo "Bitte einmalig auf dem LOGIN-NODE ausführen:"
    echo "  bash $LSDF_CODE/prepare_workspace.sh"
    exit 1
fi
WS=$(ws_find llm_run)

if [ ! -f "$WS/vllm.sif" ] || [ ! -d "$WS/$MODEL_REL" ]; then
    echo "FEHLER: Modell/Container fehlen im Workspace ($WS)."
    echo "Bitte einmalig auf dem LOGIN-NODE ausführen:"
    echo "  bash $LSDF_CODE/prepare_workspace.sh"
    exit 1
fi

echo "[1/5] Workspace OK: $WS"

# ----------------------------------------------------------------------------
# 2. Code + Input vom Workspace-Cache holen
# ----------------------------------------------------------------------------
# Code liegt bei GPU-Nodes nicht per LSDF verfügbar, daher aus Workspace-Kopie.
# prepare_workspace.sh hat code/ und data/ bereits in den WS gelegt.
WS_CODE="$WS/code"
WS_DATA="$WS/data"
INPUT_PATH="$WS_DATA/$INPUT_NAME"

if [ ! -f "$INPUT_PATH" ]; then
    echo "FEHLER: Input-Datei nicht im Workspace: $INPUT_PATH"
    echo "Liegt sie in LSDF unter data/? Dann prepare_workspace.sh erneut laufen lassen."
    exit 1
fi
echo "[2/5] Input gefunden: $INPUT_PATH"

# Lokales Ergebnis (erst im Workspace, am Ende nach LSDF)
WS_RESULTS="$WS/results"
mkdir -p "$WS_RESULTS"
OUTPUT_PATH="$WS_RESULTS/$OUTPUT_NAME"

# --- Wiederaufnahme-Status ---
# prepare_workspace.sh (Login-Node) holt vorhandene Teil-Ergebnisse aus LSDF
# in den Workspace zurück. Hier prüfen wir nur ob eine Ergebnisdatei existiert.
# Die Pipeline überspringt bereits gelabelte (comment_id, task)-Paare selbst.
if [ -f "$OUTPUT_PATH" ]; then
    EXISTING_LINES=$(wc -l < "$OUTPUT_PATH")
    echo "[2b] WIEDERAUFNAHME: $OUTPUT_PATH existiert ($EXISTING_LINES Zeilen bereits gelabelt)."
    echo "     Bereits erledigte (comment_id, task)-Paare werden übersprungen."
else
    echo "[2b] NEUSTART: keine vorhandene Ergebnisdatei, labele von vorne."
fi

# ----------------------------------------------------------------------------
# 3. vLLM-Server INTERN starten (Hintergrund)
# ----------------------------------------------------------------------------
export HF_HOME="$WS/hf_cache"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CC=gcc
export CXX=g++
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_PATH="$WS/$MODEL_REL"

echo "[3/5] Starte vLLM-Server intern (Port $PORT) ..."
singularity exec --nv \
    --bind "$WS:$WS" \
    "$WS/vllm.sif" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size 4 \
        --enforce-eager \
        --max-model-len 4096 \
        --served-model-name llama-70b \
        --port $PORT \
        --host 0.0.0.0 \
    > "server_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!
echo "     Server-PID: $SERVER_PID (Log: server_${SLURM_JOB_ID}.log)"

# Sicherstellen dass der Server beim Job-Ende gekillt wird
trap "echo 'Stoppe Server...'; kill $SERVER_PID 2>/dev/null" EXIT

# ----------------------------------------------------------------------------
# 4. Pipeline laufen lassen (wartet selbst auf Server-Readiness)
# ----------------------------------------------------------------------------
echo "[4/5] Starte Label-Pipeline ..."
singularity exec --nv \
    --bind "$WS:$WS" \
    "$WS/vllm.sif" \
    python3 "$WS_CODE/label_pipeline.py" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --base-url "$BASE_URL" \
        --model llama-70b \
        --wait-server 900

PIPELINE_RC=$?
echo "     Pipeline beendet mit Code $PIPELINE_RC"

# ----------------------------------------------------------------------------
# 5. Ergebnisse nach LSDF zurückschreiben
# ----------------------------------------------------------------------------
# ACHTUNG: LSDF ist auf GPU-Nodes nicht gemountet! Deshalb schreiben wir NICHT
# direkt nach LSDF, sondern lassen das Ergebnis im Workspace. Ein separater
# Login-Node-Schritt (sync_results_to_lsdf.sh) kopiert es nach LSDF.
# Wir legen aber eine Marker-Datei an, damit man weiß was zu syncen ist.
echo "[5/5] Ergebnis liegt im Workspace: $OUTPUT_PATH"
echo "$OUTPUT_NAME" >> "$WS_RESULTS/.to_sync"
echo ""
echo "============================================================"
echo "FERTIG. Nächster Schritt auf dem LOGIN-NODE:"
echo "  bash $LSDF_CODE/sync_results_to_lsdf.sh"
echo "  (kopiert Ergebnisse aus dem Workspace nach LSDF)"
echo "============================================================"

exit $PIPELINE_RC
