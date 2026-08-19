#!/bin/bash
# =============================================================================
# sync_results_to_lsdf.sh  -  AUF DEM LOGIN-NODE ausführen
# -----------------------------------------------------------------------------
# Kopiert die Labeling-Ergebnisse aus dem Workspace nach LSDF (dauerhaft).
# Muss auf dem LOGIN-NODE laufen (LSDF nur dort verfügbar).
#
# Aufruf:
#   bash sync_results_to_lsdf.sh
# =============================================================================

set -euo pipefail

LSDF_PROJECT=/lsdf/kit/itz/projects/delib_lab
LSDF_RESULTS=$LSDF_PROJECT/results

WS=$(ws_find llm_run)
WS_RESULTS="$WS/results"

if [ ! -d "$WS_RESULTS" ]; then
    echo "Keine Ergebnisse im Workspace gefunden ($WS_RESULTS)."
    exit 0
fi

echo "=== Kopiere Ergebnisse nach LSDF ==="
mkdir -p "$LSDF_RESULTS"
rsync -av "$WS_RESULTS/"*.ndjson "$LSDF_RESULTS/" 2>/dev/null || {
    echo "Keine .ndjson-Dateien zum Kopieren."
    exit 0
}

# Gruppenrechte setzen
chmod -R g+rX "$LSDF_RESULTS" 2>/dev/null || true

echo ""
echo "=== FERTIG ==="
echo "Ergebnisse liegen in LSDF: $LSDF_RESULTS"
ls -lh "$LSDF_RESULTS/"

# Sync-Marker aufräumen
rm -f "$WS_RESULTS/.to_sync"
