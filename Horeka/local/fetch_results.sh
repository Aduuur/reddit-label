#!/bin/bash
# =============================================================================
# fetch_results.sh  -  AUF DEM LOKALEN MAC ausführen
# -----------------------------------------------------------------------------
# Holt die Labeling-Ergebnisse von LSDF (via HoreKa Login-Node) auf den
# lokalen Rechner.
#
# Voraussetzung: sync_results_to_lsdf.sh wurde auf HoreKa ausgeführt,
# sodass die Ergebnisse in LSDF liegen.
#
# Aufruf:
#   bash fetch_results.sh
#   bash fetch_results.sh /eigener/zielordner
# =============================================================================

set -euo pipefail

# ---- Konfiguration (ggf. anpassen) ----
HOREKA_USER="unoim"
HOREKA_HOST="horeka.scc.kit.edu"
LSDF_RESULTS="/lsdf/kit/itz/projects/delib_lab/results"

# Zielordner lokal (Default: ./results)
LOCAL_DIR="${1:-./results}"
mkdir -p "$LOCAL_DIR"

echo "=== Hole Ergebnisse von LSDF ==="
echo "  Von:  ${HOREKA_USER}@${HOREKA_HOST}:${LSDF_RESULTS}/"
echo "  Nach: $LOCAL_DIR/"
echo ""

# rsync über SSH (nur neue/geänderte Dateien)
rsync -avz --progress \
    "${HOREKA_USER}@${HOREKA_HOST}:${LSDF_RESULTS}/*.ndjson" \
    "$LOCAL_DIR/"

echo ""
echo "=== FERTIG ==="
echo "Ergebnisse liegen in: $LOCAL_DIR/"
ls -lh "$LOCAL_DIR/"
