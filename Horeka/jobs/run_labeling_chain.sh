#!/bin/bash
# =============================================================================
# run_labeling_chain.sh  -  Selbst-fortsetzender Labeling-Job
# -----------------------------------------------------------------------------
# Wie run_labeling_async.sh, aber reiht am Ende AUTOMATISCH einen Folgejob ein,
# falls noch nicht alle Kommentare fertig gelabelt sind. So läuft das Labeling
# über mehrere 8h-Jobs hinweg ohne manuelles Nachstarten.
#
# Die Kette stoppt automatisch, sobald alle (comment_id, task)-Paare erledigt
# sind (Vergleich Anzahl erwartet vs. Anzahl in der Ergebnisdatei).
#
# Aufruf (einmal starten, Rest läuft automatisch):
#   sbatch run_labeling_chain.sh <INPUTNAME> <CONCURRENCY>
# Beispiel:
#   sbatch run_labeling_chain.sh conversation_threads_flat.ndjson 48
#
# Kette abbrechen: scancel <JOBID> des gerade laufenden/wartenden Jobs.
# =============================================================================
#SBATCH --job-name=label_chain
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=chain_%j.out

set -uo pipefail

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
LSDF_PROJECT=/lsdf/kit/itz/projects/delib_lab
MODEL_REL=hf_cache/models/meta-llama/Llama-3.1-70B-Instruct

INPUT_NAME="${1:-conversation_threads_flat.ndjson}"
CONCURRENCY="${2:-48}"
OUTPUT_NAME="labels_${INPUT_NAME%.*}.ndjson"

# Anzahl Tasks (muss zur Pipeline passen!). Wird für die Vollständigkeits-
# Prüfung gebraucht: erwartete Zeilen = Kommentare * TASKS_PER_COMMENT.
TASKS_PER_COMMENT=7

PORT=8000
BASE_URL="http://localhost:${PORT}"

echo "============================================================"
echo "LABELING CHAIN-JOB   $(date)"
echo "  Node:        $(hostname)"
echo "  Input:       $INPUT_NAME"
echo "  Output:      $OUTPUT_NAME"
echo "  Concurrency: $CONCURRENCY"
echo "  Job-ID:      ${SLURM_JOB_ID}"
echo "============================================================"

# ----------------------------------------------------------------------------
# 1. Workspace-Checks
# ----------------------------------------------------------------------------
if ! ws_find llm_run >/dev/null 2>&1; then
    echo "FEHLER: Workspace 'llm_run' fehlt. Erst prepare_workspace.sh (Login-Node)."
    exit 1
fi
WS=$(ws_find llm_run)

if [ ! -f "$WS/vllm.sif" ] || [ ! -d "$WS/$MODEL_REL" ]; then
    echo "FEHLER: Modell/Container fehlen im Workspace. Erst prepare_workspace.sh."
    exit 1
fi

WS_CODE="$WS/code"
WS_DATA="$WS/data"
WS_RESULTS="$WS/results"
INPUT_PATH="$WS_DATA/$INPUT_NAME"
OUTPUT_PATH="$WS_RESULTS/$OUTPUT_NAME"
mkdir -p "$WS_RESULTS"

if [ ! -f "$INPUT_PATH" ]; then
    echo "FEHLER: Input nicht im Workspace: $INPUT_PATH"
    exit 1
fi
echo "[1] Workspace OK: $WS"

# ----------------------------------------------------------------------------
# 2. Anzahl zu labelnder Kommentare bestimmen (nicht-leere Zeilen)
# ----------------------------------------------------------------------------
# Wir zählen die Input-Zeilen (= Kommentare). Erwartete Ergebniszeilen =
# Kommentare * TASKS_PER_COMMENT. Das ist die Zielmarke für "fertig".
N_COMMENTS=$(grep -cve '^[[:space:]]*$' "$INPUT_PATH")
EXPECTED_LINES=$(( N_COMMENTS * TASKS_PER_COMMENT ))
echo "[2] Input: $N_COMMENTS Kommentare -> erwartet $EXPECTED_LINES Ergebniszeilen"

# Aktueller Stand
if [ -f "$OUTPUT_PATH" ]; then
    CURRENT_LINES=$(grep -cve '^[[:space:]]*$' "$OUTPUT_PATH")
else
    CURRENT_LINES=0
fi
echo "    Aktuell gelabelt: $CURRENT_LINES Zeilen"

# ----------------------------------------------------------------------------
# 3. vLLM-Server intern starten
# ----------------------------------------------------------------------------
export HF_HOME="$WS/hf_cache"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export CC=gcc
export CXX=g++
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_PATH="$WS/$MODEL_REL"

echo "[3] Starte vLLM-Server intern (Port $PORT) ..."
singularity exec --nv --bind "$WS:$WS" "$WS/vllm.sif" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size 4 \
        --enforce-eager \
        --max-model-len 4096 \
        --served-model-name llama-70b \
        --port $PORT --host 0.0.0.0 \
    > "server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
echo "    Server-PID: $SERVER_PID"

# ----------------------------------------------------------------------------
# 4. Pipeline laufen lassen
# ----------------------------------------------------------------------------
echo "[4] Starte Label-Pipeline (async, concurrency=$CONCURRENCY) ..."
singularity exec --nv --bind "$WS:$WS" "$WS/vllm.sif" \
    python3 "$WS_CODE/label_pipeline_async.py" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --base-url "$BASE_URL" \
        --model llama-70b \
        --concurrency "$CONCURRENCY" \
        --wait-server 900
PIPELINE_RC=$?
echo "[4] Pipeline beendet (RC=$PIPELINE_RC)"

# Server stoppen (trap greift auch, aber explizit ist sauberer)
kill $SERVER_PID 2>/dev/null
sleep 5

# ----------------------------------------------------------------------------
# 5. Vollständigkeit prüfen und ggf. Folgejob einreihen
# ----------------------------------------------------------------------------
if [ -f "$OUTPUT_PATH" ]; then
    NEW_LINES=$(grep -cve '^[[:space:]]*$' "$OUTPUT_PATH")
else
    NEW_LINES=0
fi
echo "[5] Nach diesem Job: $NEW_LINES / $EXPECTED_LINES Zeilen"

if [ "$NEW_LINES" -ge "$EXPECTED_LINES" ]; then
    echo "============================================================"
    echo "KETTE FERTIG: Alle $EXPECTED_LINES Zeilen gelabelt."
    echo "Nächster Schritt (Login-Node):"
    echo "  bash $WS_CODE/sync_results_to_lsdf.sh"
    echo "============================================================"
    exit 0
fi

# Fortschritts-Sicherung: Wenn dieser Job NICHTS Neues geschafft hat, brechen
# wir die Kette ab, um Endlosschleifen bei einem echten Fehler zu vermeiden.
if [ "$NEW_LINES" -le "$CURRENT_LINES" ]; then
    echo "============================================================"
    echo "ABBRUCH DER KETTE: Kein Fortschritt in diesem Job"
    echo "  (vorher $CURRENT_LINES, jetzt $NEW_LINES Zeilen)."
    echo "Bitte server_${SLURM_JOB_ID}.log und chain_${SLURM_JOB_ID}.out prüfen."
    echo "============================================================"
    exit 1
fi

# Noch Arbeit übrig UND Fortschritt gemacht -> Folgejob einreihen.
# Der Folgejob startet erst wenn dieser hier fertig ist (afterany).
REMAINING=$(( EXPECTED_LINES - NEW_LINES ))
echo "[5] Noch $REMAINING Zeilen offen -> reihe Folgejob ein."

NEXT_JOB=$(sbatch --parsable \
    --dependency=afterany:${SLURM_JOB_ID} \
    --job-name=label_chain \
    "$WS_CODE/run_labeling_chain.sh" "$INPUT_NAME" "$CONCURRENCY")

echo "============================================================"
echo "Folgejob eingereiht: $NEXT_JOB (startet nach diesem Job)"
echo "Kette abbrechen: scancel $NEXT_JOB"
echo "============================================================"
exit 0
