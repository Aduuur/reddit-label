#!/bin/bash
# =============================================================================
# prepare_workspace.sh  -  AUF DEM LOGIN-NODE ausführen
# -----------------------------------------------------------------------------
# Befüllt den persönlichen Workspace 'llm_run' mit allem was der Labeling-Job
# braucht: Modell, Container, Code und Input-Daten - alles aus LSDF.
#
# Muss auf dem LOGIN-NODE laufen, weil nur dort LSDF verfügbar ist.
# Läuft am besten via nohup im Hintergrund (Modell-Kopie dauert).
#
# Aufruf:
#   nohup bash prepare_workspace.sh > prepare.log 2>&1 &
#   tail -f prepare.log
# =============================================================================

set -euo pipefail

LSDF_PROJECT=/lsdf/kit/itz/projects/delib_lab
MODEL_REL=hf_cache/models/meta-llama/Llama-3.1-70B-Instruct

echo "=== Workspace anlegen (falls nötig) ==="
if ws_find llm_run >/dev/null 2>&1; then
    echo "Workspace existiert: $(ws_find llm_run)"
else
    ws_allocate llm_run 30
fi
WS=$(ws_find llm_run)
echo "Workspace: $WS"

echo "=== Container kopieren ==="
if [ ! -f "$WS/vllm.sif" ]; then
    cp "$LSDF_PROJECT/vllm.sif" "$WS/"
    echo "Container kopiert."
else
    echo "Container schon vorhanden."
fi

# Modelle in den Workspace kopieren. Standardmäßig werden alle Modelle unter
# hf_cache/models/ mitgenommen, die in LSDF liegen (Llama, Qwen, ...). So ist
# im Workspace immer jedes Modell verfügbar, egal mit welchem du labeln willst.
# Mit MODELS_TO_SYNC kann man die Auswahl einschränken (Leerzeichen-getrennte
# relative Pfade), z.B.:
#   MODELS_TO_SYNC="hf_cache/models/Qwen/Qwen2.5-72B-Instruct" bash prepare_workspace.sh
if [ -n "${MODELS_TO_SYNC:-}" ]; then
    MODEL_LIST="$MODELS_TO_SYNC"
else
    # alle Modell-Ordner unter hf_cache/models/<org>/<name> automatisch finden
    MODEL_LIST=$(cd "$LSDF_PROJECT" && find hf_cache/models -mindepth 2 -maxdepth 2 -type d 2>/dev/null)
fi

for MREL in $MODEL_LIST; do
    if [ -d "$LSDF_PROJECT/$MREL" ]; then
        echo "=== Modell kopieren: $MREL ==="
        mkdir -p "$WS/$MREL"
        rsync -a --info=progress2 "$LSDF_PROJECT/$MREL/" "$WS/$MREL/"
    else
        echo "WARNUNG: Modell $MREL nicht in LSDF gefunden, übersprungen."
    fi
done

echo "=== Code kopieren ==="
mkdir -p "$WS/code"
rsync -a "$LSDF_PROJECT/code/" "$WS/code/"

echo "=== Input-Daten kopieren ==="
mkdir -p "$WS/data"
if [ -d "$LSDF_PROJECT/data" ]; then
    rsync -a "$LSDF_PROJECT/data/" "$WS/data/"
    echo "Daten kopiert:"
    ls -lh "$WS/data/"
else
    echo "WARNUNG: $LSDF_PROJECT/data existiert nicht. Bitte Input-Daten dorthin legen."
fi

echo "=== Vorhandene Ergebnisse aus LSDF zurückholen (für Wiederaufnahme) ==="
# Wenn schon Teil-Ergebnisse in LSDF liegen, kopieren wir sie in den Workspace,
# damit ein erneuter Labeling-Job dort nahtlos weitermacht statt neu zu starten.
mkdir -p "$WS/results"
if [ -d "$LSDF_PROJECT/results" ]; then
    rsync -a "$LSDF_PROJECT/results/"*.ndjson "$WS/results/" 2>/dev/null && \
        echo "Vorhandene Ergebnisse zurückgeholt:" && ls -lh "$WS/results/" || \
        echo "Keine vorhandenen Ergebnisse in LSDF (frischer Start)."
else
    echo "Noch keine Ergebnisse in LSDF (frischer Start)."
fi

echo ""
echo "=== FERTIG ==="
echo "Workspace bereit: $WS"
echo "Nächster Schritt: sbatch run_labeling.sh <INPUTNAME>"
