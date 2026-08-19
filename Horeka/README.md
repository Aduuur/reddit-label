# Reddit Discourse Labeling auf HoreKa

LLM-basiertes Annotieren von Reddit-Diskursdaten mit **Llama 3.1 70B** auf dem
KIT-Hochleistungsrechner **HoreKa**. Jeder Kommentar wird über sieben
Diskurs-Dimensionen gelabelt; zusätzlich werden argumentative Einheiten
(claim + proof) extrahiert.

Projekt: `redditlabel` / Batzdorfer · LSDF-Gruppe `KIT-redditlabel-LSDF`

---

## Was das System macht

Ein einzelner SLURM-Job startet auf einem GPU-Knoten intern einen vLLM-Server,
labelt alle Kommentare gegen `localhost` (kein SSH-Tunnel) und schreibt die
Ergebnisse fortlaufend als NDJSON. Modell, Container, Code und Daten liegen
dauerhaft auf LSDF; für den Lauf werden sie in einen HoreKa-Workspace kopiert
(LSDF ist auf GPU-Knoten nicht gemountet).

### Die sieben Dimensionen

| Dimension | Bereich | Bedeutung |
|---|---|---|
| `stance_intensity` | 1–6 | Haltung zum Thema (1 = stark dagegen, 6 = stark dafür) |
| `epistemic_modality` | 0–1 | Ausgedrückte Unsicherheit / Hedging |
| `justification_density` | ≥ 0 | Begründungsdichte pro 100 Wörter |
| `responsiveness` | 0–1 | Bezug auf den Eltern-Kommentar |
| `agreement` | −1…1 | Zustimmung zum Eltern-Kommentar |
| `civility` | 1–6 | Höflichkeit (1 = beleidigend, 6 = sehr höflich) |
| `sarcasm` | 0/1 | Sarkasmus / Ironie |

---

## Repository-Struktur

```
.
├── pipeline/
│   ├── prompts.py                 # Task-Definitionen + System-Prompts
│   ├── data_io.py                 # Daten-IO (pandas-frei) + Längen-Check
│   ├── label_pipeline_async.py    # Parallele Pipeline (Vollbetrieb)
│   └── label_pipeline.py          # Serielle Pipeline (Debug)
│
├── preprocessing/
│   └── resolve_predecessor.py     # Eltern-Text über parent_id auflösen
│
├── jobs/
│   ├── run_labeling_chain.sh      # Selbst-fortsetzende Kette (Standard)
│   ├── run_labeling_async.sh      # Ein paralleler Job
│   └── run_labeling.sh            # Ein serieller Job (Debug)
│
├── login_node/
│   ├── prepare_workspace.sh       # Workspace aus LSDF befüllen
│   └── sync_results_to_lsdf.sh    # Ergebnisse zurück nach LSDF
│
├── local/
│   └── fetch_results.sh           # Ergebnisse auf den eigenen Rechner holen
│
├── analysis/
│   └── make_stats_svg.py          # Übersichts-Diagramme (SVG, ohne Extra-Pakete)
│
├── docs/
│   ├── README_LABELING.md         # Technische Detail-Anleitung
│   └── Team_Anleitung_HoreKa_Labeling.docx  # Einsteiger-Anleitung fürs Team
│
└── README.md                      # diese Datei
```

> Hinweis: In LSDF liegen die Skripte flach in `code/` (kein Unterordner).
> Die Ordnerstruktur hier dient nur der Übersicht im Repo. Beim Deploy nach
> LSDF werden die Dateien aus `pipeline/`, `preprocessing/`, `jobs/` und
> `login_node/` gemeinsam nach `code/` kopiert.

---

## Datenformat

Eingabe als `.ndjson` (empfohlen) oder `.xlsx`. Erwartete Felder:

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `comment_id` | ja | Eindeutige ID |
| `body` | ja | Kommentartext |
| `predecessor` | optional | Text des Eltern-Kommentars |

Die Reddit-Rohdaten enthalten statt `predecessor` nur eine `parent_id`. Das
Skript `resolve_predecessor.py` löst daraus den Eltern-Text auf (siehe unten).

---

## Ablauf (Kurzfassung)

### 0. Einmalig: Workspace einrichten (Login-Node)
```bash
cp /lsdf/kit/itz/projects/delib_lab/code/prepare_workspace.sh ~/
nohup bash ~/prepare_workspace.sh > ~/prepare.log 2>&1 &
tail -f ~/prepare.log        # warten bis === FERTIG ===
```

### 1. Eltern-Text auflösen (Login-Node, pure Python)
```bash
cd /lsdf/kit/itz/projects/delib_lab
python3 code/resolve_predecessor.py \
  --input  data/conversation_threads_flat.ndjson \
  --output data/conversation_threads_resolved.ndjson
```
Die Statistik zeigt, wie viele Kommentare einen Eltern-Text bekommen haben.

### 2. Workspace aktualisieren (Login-Node)
```bash
bash ~/prepare_workspace.sh
```

### 3. Labeling starten (Login-Node)
```bash
cd $(ws_find llm_run)/code
# Standard: selbst-fortsetzende Kette, Concurrency 48
sbatch run_labeling_chain.sh conversation_threads_resolved.ndjson 48
```

### 4. Beobachten
```bash
squeue -u $USER
tail -f chain_<JOBID>.out     # zeigt calls/s und ETA
```

### 5. Sichern + abholen
```bash
bash $(ws_find llm_run)/code/sync_results_to_lsdf.sh   # Login-Node
# auf dem eigenen Rechner:
scp DEINACCOUNT@horeka.scc.kit.edu:/lsdf/kit/itz/projects/delib_lab/results/*.ndjson .
```

Ausführliche Anleitung: `docs/README_LABELING.md`
Einsteiger-Anleitung fürs Team: `docs/Team_Anleitung_HoreKa_Labeling.docx`

---

## ABSTAIN-Logik

Das System vergibt einen Wert für jede Dimension und drückt Unsicherheit über
das `confidence`-Feld aus. `ABSTAIN` wird nur in zwei klar definierten Fällen
gesetzt – deterministisch im Code, nicht durch das Modell:

1. **Zu kurzer Text** (`TOO_SHORT`): unter `MIN_TOKENS` Tokens (Default 15,
   in `data_io.py` einstellbar) → alle Tasks ABSTAIN, ohne Modell-Aufruf.
2. **Kein Eltern-Kommentar** (`NO_PARENT`): `responsiveness` und `agreement`
   ohne `predecessor` → ABSTAIN (fehlende Grundlage).

In allen anderen Fällen gibt das Modell einen Wert plus ehrliche Konfidenz.
Filtern nach Unsicherheit passiert nachgelagert über die Konfidenz-Schwelle.

> **Kalibrierungshinweis:** LLM-Konfidenzwerte sind nicht automatisch gut
> kalibriert. Vor dem Verlassen auf absolute Schwellen (z. B. „alles > 0.7
> ist verlässlich") sollte die Konfidenz an einem kleinen, handgelabelten
> Gold-Set geprüft werden. Die relative Ordnung (hoch vs. niedrig) ist meist
> brauchbar; die absoluten Werte nicht ohne Prüfung.

---

## Wiederaufnahme

Der Output-Name ist fest pro Eingabedatei (`labels_<name>.ndjson`). Die
Pipeline überspringt bereits gelabelte `(comment_id, task)`-Paare. Ein
abgebrochener Lauf wird durch erneutes `sbatch` desselben Befehls nahtlos
fortgesetzt. Die Kette (`run_labeling_chain.sh`) reiht Folgejobs automatisch
ein, bis alles fertig ist.

---

## Infrastruktur-Notizen

- **Modell:** `meta-llama/Llama-3.1-70B-Instruct` (~132 GB, 30 safetensors)
- **Server:** vLLM, tensor-parallel über 4 GPUs, `--max-model-len 4096`,
  `--enforce-eager`, FlashInfer-Sampler deaktiviert (HoreKa-Kompatibilität)
- **LSDF-Projektpfad:** `/lsdf/kit/itz/projects/delib_lab/`
- **HoreKa-Projekt:** `hk-project-starter-p0026357`
- **Workspace:** `llm_run` (läuft nach 30 Tagen ab: `ws_extend llm_run 30`)

---

## Datenschutz

Die LSDF-Nutzungsbedingungen schließen personenbezogene Daten aus. Reddit-Daten
mit Klarnamen oder Usernames sind je nach Auslegung betroffen. Vor einem
Vollbetrieb mit der Projektleitung (Veronika Batzdorfer) klären.

---

## Kontakt

Arthur Rolf · KIT · bei Systemfragen: hpc-support@scc.kit.edu / bw-support.scc.kit.edu
