# Reddit Discourse Labeling auf HoreKa

LLM-basiertes Annotieren von Reddit-Diskursdaten mit **Llama 3.1 70B** (und
optional **Qwen2.5-72B** zum Vergleich) auf dem KIT-Hochleistungsrechner
**HoreKa**. Jeder Kommentar wird über sieben Diskurs-Dimensionen gelabelt;
zusätzlich werden argumentative Einheiten (claim + proof) extrahiert.

Projekt: `redditlabel` / Batzdorfer · LSDF-Gruppe `KIT-redditlabel-LSDF`

Ausführliche Schritt-für-Schritt-Anleitung mit allen Befehlen und
Anpassungshinweisen: **`docs/Team_Anleitung_HoreKa_Labeling.docx`**.

---

## Was das System macht

Ein einzelner SLURM-Job startet auf einem GPU-Knoten intern einen vLLM-Server,
labelt alle Kommentare gegen `localhost` (kein SSH-Tunnel) und schreibt die
Ergebnisse fortlaufend als NDJSON. Modell, Container, Code und Daten liegen
dauerhaft in LSDF; für den Lauf werden sie in einen HoreKa-Workspace kopiert
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
│   ├── data_io.py                 # Daten-IO (pandas-frei), Token-Zaehlung
│   ├── label_pipeline_async.py    # Parallele Pipeline (Vollbetrieb)
│   ├── label_pipeline.py          # Serielle Pipeline (Debug)
│   └── label_rationale.py         # Test-Pipeline mit Modell-Begruendung (CoT)
│
├── preprocessing/
│   ├── build_threads_colab.py     # Threads aus Submissions+Kommentaren bauen (Colab)
│   ├── resolve_predecessor.py     # Eltern-Text ueber parent_id aufloesen
│   └── sample_threads.py          # Vollstaendige Threads bis Ziel-Anzahl auswaehlen
│
├── jobs/
│   ├── run_labeling_chain.sh      # Selbst-fortsetzende Kette (Standard)
│   ├── run_labeling_async.sh      # Ein paralleler Job (modellwaehlbar)
│   ├── run_labeling.sh            # Ein serieller Job (Debug)
│   └── run_rationale.sh           # Job fuer die Begruendungs-Test-Pipeline
│
├── login_node/
│   ├── prepare_workspace.sh       # Workspace aus LSDF befuellen (alle Modelle)
│   ├── sync_results_to_lsdf.sh    # Ergebnisse zurueck nach LSDF
│   └── download_model.py          # Modell aus Hugging Face nach LSDF laden
│
├── local/
│   └── fetch_results.sh           # Ergebnisse auf den eigenen Rechner holen
│
├── analysis/
│   ├── make_stats_svg.py          # Uebersichts-Diagramme (SVG, ohne Extra-Pakete)
│   ├── reddit_datenanalyse.ipynb  # Explorative Analyse der EINGABEdaten (lokal)
│   └── reddit_ergebnis_analyse.ipynb  # Analyse der LABEL-Ergebnisse (lokal)
│
├── docs/
│   ├── README_LABELING.md         # Technische Detail-Anleitung
│   └── Team_Anleitung_HoreKa_Labeling.docx  # Vollstaendige Einsteiger-Anleitung
│
└── README.md                      # diese Datei
```

> In LSDF liegen die Skripte flach in `code/`. Die Ordnerstruktur hier dient nur
> der Uebersicht im Repo.

---

## Die drei Orte

Fast alle Anfaengerfehler kommen daher, dass diese drei Orte verwechselt werden:

- **Dein Rechner** (Mac/Laptop): Daten hochladen, Ergebnisse herunterladen, Skripte bearbeiten.
- **HoreKa**: das eigentliche Rechnen. Besteht aus Login-Node (Vorbereiten, sieht LSDF) und GPU-Nodes (Rechnen, sehen LSDF NICHT).
- **LSDF**: dauerhafter, gemeinsamer Projektspeicher. Fuer alle im Team gleich erreichbar.

**Merksatz:** LSDF ist nur auf dem Login-Node sichtbar, nicht auf den GPU-Nodes.
Deshalb kopiert `prepare_workspace.sh` alles einmal in den (schnellen, aber
privaten und nach 30 Tagen ablaufenden) Workspace, von dem der GPU-Node liest.

**Regel fuer die Zusammenarbeit:** Wer einen Lauf gemacht hat, muss die Ergebnisse
einmal mit `sync_results_to_lsdf.sh` von seinem Workspace nach LSDF kopieren.
Erst dann koennen andere sie herunterladen - Workspaces sind privat.

---

## Datenformat

Eingabe als `.ndjson` (empfohlen) oder `.xlsx`. Pflichtfelder je Kommentar:
`comment_id` und `body`. Optional `predecessor` (Text des Eltern-Kommentars);
den ergaenzt `resolve_predecessor.py` automatisch.

Der Post selbst wird als Wurzelknoten mitgefuehrt (Feld `is_post`, `depth = -1`),
damit auch die obersten Kommentare einen Vorgaenger haben.

---

## Ablauf (Kurzfassung)

Platzhalter anpassen: `EUERACCOUNT` = dein KIT-Kurzname,
`DATEINAME` = deine Eingabedatei.

### 0. Einmalig: Workspace einrichten (Login-Node)
```bash
cp /lsdf/kit/itz/projects/delib_lab/code/prepare_workspace.sh ~/
nohup bash ~/prepare_workspace.sh > ~/prepare.log 2>&1 &
tail -f ~/prepare.log        # warten bis === FERTIG ===
```

### 1. Daten hochladen (dein Rechner)
```bash
scp DATEINAME.ndjson EUERACCOUNT@horeka.scc.kit.edu:/lsdf/kit/itz/projects/delib_lab/data/
```

### 2. Eltern-Text aufloesen (Login-Node, pure Python)
```bash
cd /lsdf/kit/itz/projects/delib_lab
python3 code/resolve_predecessor.py \
  --input  data/DATEINAME.ndjson \
  --output data/DATEINAME_resolved.ndjson
```

### 3. Workspace aktualisieren (Login-Node)
```bash
bash ~/prepare_workspace.sh
```

### 4. Labeling starten (Login-Node)
```bash
cd $(ws_find llm_run)/code
sbatch run_labeling_chain.sh DATEINAME_resolved.ndjson 48
```

### 5. Beobachten
```bash
squeue -u $USER
tail -f chain_<JOBID>.out     # zeigt calls/s und ETA
```

### 6. Sichern (die Person, die gelabelt hat) + herunterladen (dein Rechner)
```bash
# Login-Node:
bash $(ws_find llm_run)/code/sync_results_to_lsdf.sh
# dein Rechner:
scp EUERACCOUNT@horeka.scc.kit.edu:/lsdf/kit/itz/projects/delib_lab/results/labels_*_DATEINAME_resolved.ndjson .
```

---

## ABSTAIN-Logik

ABSTAIN (Enthaltung) gibt es nur noch in **einem** Fall: bei `responsiveness`
und `agreement`, wenn kein Eltern-Kommentar vorhanden ist (`NO_PARENT`). In
allen anderen Faellen gibt das Modell immer einen Wert; Unsicherheit drueckt es
ueber die Konfidenz aus. Jede Ergebniszeile enthaelt `n_tokens`, sodass die Laenge
nachgelagert gefiltert werden kann, statt ueber ABSTAIN.

> **Offener Punkt (Stand jetzt):** Bei `agreement` liefert das Modell in einem
> kleinen Teil der Faelle `{"task": "ABSTAIN", ...}`, obwohl ein Parent da ist.
> Wie damit umzugehen ist (ABSTAIN zulassen vs. Wert erzwingen), ist im Team
> noch abzustimmen. Die Rohantwort wird bei Fehlern mitgespeichert.

> **Kalibrierungshinweis:** LLM-Konfidenzwerte sind oft ueberkonfident. Sie taugen
> zum Aussortieren der klar unsicheren Faelle, aber nicht als feine
> Praezisionsskala. Fuer belastbare Schwellen: an einem handgelabelten Gold-Set pruefen.

---

## Zweites Modell (Vergleich)

Ein zweites Modell (z. B. Qwen2.5-72B-Instruct) laesst sich zum Vergleich
hosten. Download nach LSDF:
```bash
# Login-Node, ohne Container (huggingface_hub muss verfuegbar sein):
python3 download_model.py --model Qwen/Qwen2.5-72B-Instruct
chmod -R g+rX /lsdf/kit/itz/projects/delib_lab/hf_cache/models/Qwen/
```
`prepare_workspace.sh` kopiert automatisch alle Modelle unter
`hf_cache/models/` in den Workspace. Labeln mit dem zweiten Modell:
```bash
MODEL_REL=hf_cache/models/Qwen/Qwen2.5-72B-Instruct MODEL_NAME=qwen-72b \
  sbatch run_labeling_async.sh DATEINAME_resolved.ndjson 48
```
Der Ergebnis-Dateiname enthaelt den Modellnamen (`labels_qwen-72b_...` vs.
`labels_llama-70b_...`), sodass sich die Laeufe sauber vergleichen lassen.
Wichtig fuer Fairness: gleicher Eingabedatensatz, gleiche Prompts.

---

## Modell-Begruendung (Test)

`label_rationale.py` (per `run_rationale.sh`) ist eine Test-Variante, bei der
das Modell zuerst in einem Satz begruendet, welche Textstelle ausschlaggebend
war, und dann das JSON ausgibt (Chain-of-Thought). Jede Ergebniszeile bekommt
ein Feld `rationale`. Nur fuer kleine Test-Datensaetze gedacht:
```bash
sbatch run_rationale.sh test_threads_500_resolved.ndjson 50
```

---

## Wiederaufnahme & robuste Kette

Der Output-Name ist fest pro Eingabedatei und Modell; die Pipeline ueberspringt
bereits gelabelte `(comment_id, task)`-Paare. Ein abgebrochener Lauf wird durch
erneutes `sbatch` nahtlos fortgesetzt.

Die Kette (`run_labeling_chain.sh`) reiht das naechste Kettenglied **vorab** ein
(mit `afterany`-Abhaengigkeit), damit sie auch dann weiterlaeuft, wenn ein Job an
die 8-Stunden-Grenze stoesst. Nach `squeue -u $USER` sollten daher zwei Jobs
sichtbar sein (laufend + wartend mit `Dependency`). Ein Stall-Zaehler bricht ab,
wenn zwei Jobs in Folge keinen Fortschritt machen.

---

## Infrastruktur-Notizen

- **Modelle:** `meta-llama/Llama-3.1-70B-Instruct` (~132 GB), optional
  `Qwen/Qwen2.5-72B-Instruct` (~145 GB)
- **Server:** vLLM, tensor-parallel ueber 4 GPUs, `--max-model-len 4096`,
  `--enforce-eager`, FlashInfer-Sampler deaktiviert (HoreKa-Kompatibilitaet)
- **LSDF-Projektpfad:** `/lsdf/kit/itz/projects/delib_lab/`
- **HoreKa-Projekt:** `hk-project-starter-p0026357`
- **Workspace:** `llm_run` (laeuft nach 30 Tagen ab: `ws_extend llm_run 30`)

---

## Datenschutz

Die LSDF-Nutzungsbedingungen schliessen personenbezogene Daten aus. Reddit-Daten
mit Usernames sind je nach Auslegung betroffen. Vor einem Vollbetrieb mit der
Projektleitung (Veronika Batzdorfer) klaeren. Falls die KI-Toolbox des SCC genutzt
wird: fuer personenbezogene Daten nur die lokalen Modelle, nicht die Cloud-Modelle.

---

## Kontakt

Arthur Rolf · KIT · bei Systemfragen: hpc-support@scc.kit.edu / bw-support.scc.kit.edu
