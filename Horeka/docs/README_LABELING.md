# HPC-Labeling – komplett auf HoreKa

Labeling-Pipeline die vollständig auf HoreKa läuft. Kein lokales Datenhandling,
kein SSH-Tunnel. Server und Pipeline laufen im selben SLURM-Job auf demselben
GPU-Node und kommunizieren über `localhost`.

---

## Wichtiges Architektur-Prinzip

**LSDF ist nur auf Login-Nodes verfügbar, nicht auf GPU-Compute-Nodes.**

Deshalb der zweistufige Aufbau:
- **Login-Node** (hat LSDF): Daten/Modell zwischen LSDF und Workspace bewegen
- **GPU-Node** (hat kein LSDF): rechnet nur mit der Workspace-Kopie

```
                LOGIN-NODE                        GPU-NODE (SLURM-Job)
   LSDF  <──rsync──>  Workspace(llm_run)  ──────>  liest Modell+Daten
   (Master)           (Arbeitskopie)               startet Server intern
                                                    labelt via localhost
                                                    schreibt Ergebnis
   LSDF  <──rsync──   Workspace(results)  <──────  Ergebnis
```

---

## Verzeichnisstruktur in LSDF

```
/lsdf/kit/itz/projects/delib_lab/
├── vllm.sif                  # Container (Master)
├── hf_cache/                 # Modell (Master)
├── code/                     # Pipeline-Code
│   ├── label_pipeline.py
│   ├── prompts.py
│   ├── run_labeling.sh
│   ├── prepare_workspace.sh
│   └── sync_results_to_lsdf.sh
├── data/                     # Input-Daten (hier ablegen!)
│   └── conversation_threads_flat.ndjson
└── results/                  # Ergebnisse (landen hier)
```

---

## Einmalige Einrichtung (pro Person)

Auf dem **Login-Node**:

```bash
# Workspace mit Modell/Container/Code/Daten befüllen (Modell-Kopie dauert ~1-2h)
cp /lsdf/kit/itz/projects/delib_lab/code/prepare_workspace.sh ~/
nohup bash ~/prepare_workspace.sh > prepare.log 2>&1 &
tail -f prepare.log   # Ctrl+C zum Rausgehen, läuft weiter
```

Warten bis `=== FERTIG ===`.

---

## Labeling-Durchlauf

### 1. Input-Daten nach LSDF legen (Login-Node)

```bash
# Vom Mac aus hochladen:
scp meine_daten.ndjson unoim@horeka.scc.kit.edu:/lsdf/kit/itz/projects/delib_lab/data/

# Dann Workspace aktualisieren (Login-Node):
bash ~/prepare_workspace.sh   # kopiert nur neue Daten (rsync)
```

### 2. Labeling-Job starten (Login-Node)

```bash
cd $(ws_find llm_run)/code
sbatch run_labeling.sh conversation_threads_flat.ndjson
```

Der Job:
- startet den vLLM-Server intern
- wartet bis er bereit ist
- labelt alle Zeilen (alle 7 Tasks + Argument-Extraktion)
- schreibt Ergebnis in den Workspace

### 3. Fortschritt beobachten

```bash
squeue -u $USER
tail -f label_*.out
```

### 4. Ergebnisse nach LSDF syncen (Login-Node)

```bash
bash $(ws_find llm_run)/code/sync_results_to_lsdf.sh
```

### 5. Ergebnisse lokal abholen (Mac)

```bash
bash fetch_results.sh
# oder in eigenen Ordner:
bash fetch_results.sh ~/meine_ergebnisse
```

---

## Input-Format

Unterstützt `.ndjson` und `.xlsx`. Erwartete Spalten:
- `comment_id` – eindeutige ID
- `body` – der zu labelnde Text
- `predecessor` (optional) – Eltern-Kommentar für Responsiveness/Agreement

Andere Spaltennamen? Per CLI-Flags in `run_labeling.sh` anpassbar
(`--id-col`, `--text-col`, `--parent-col`).

---

## Parallelisierung (für große Datenmengen)

Für mehrere 100.000 Kommentare gibt es die **async-Variante**, die viele
Requests gleichzeitig an den Server schickt (vLLM batcht sie intern). Das
erhöht den Durchsatz um ein Vielfaches gegenüber der seriellen Version.

**Serielle Version** (`run_labeling.sh` + `label_pipeline.py`): ein Request
nach dem anderen. Gut zum Debuggen, langsam bei viel Daten.

**Async-Version** (`run_labeling_async.sh` + `label_pipeline_async.py`):
konfigurierbare Concurrency. Empfohlen für den Vollbetrieb.

### Aufruf

```bash
cd $(ws_find llm_run)/code
# sbatch run_labeling_async.sh <INPUTNAME> <CONCURRENCY>
sbatch run_labeling_async.sh conversation_threads_flat.ndjson 48
```

Das zweite Argument ist die Anzahl gleichzeitiger Requests (Default 48).

### Concurrency wählen

| Wert   | Wann                                                        |
|--------|-------------------------------------------------------------|
| 16-32  | Konservativ, wenig GPU-Speicherdruck                        |
| 48     | Guter Startwert (Default)                                   |
| 64-128 | Aggressiv – nur wenn der Server stabil bleibt               |

**Vorgehen:** Mit 48 starten, im `label_*.out` die `calls/s` beobachten.
Steigt der Durchsatz bei höherer Concurrency nicht mehr, ist die GPU
ausgelastet – dann nicht weiter erhöhen. Bei OOM-Fehlern im
`server_*.log` (out of memory) die Concurrency senken.

### Laufzeit hochrechnen

Die async-Pipeline zeigt alle 15 Sekunden `calls/s` und eine ETA. Damit
kannst du früh abschätzen ob die Datenmenge ins 8h-Limit passt oder ob du
sie splitten musst. Beispiel: 300.000 Kommentare × 8 Calls = 2,4 Mio Calls;
bei 60 calls/s wären das ~11h → auf mehrere Jobs aufteilen (die
Wiederaufnahme macht das unkritisch).

### Wiederaufnahme

Funktioniert identisch zur seriellen Version (fester Output-Name,
Überspringen erledigter Paare). Der einzige Schreiber läuft über eine
interne Queue, sodass parallele Requests die NDJSON nicht korrumpieren.

---

## Chain-Job: automatisch über mehrere 8h-Jobs

Für sehr große Datenmengen, die nicht in ein 8h-Limit passen, gibt es
`run_labeling_chain.sh`. Der Job reiht am Ende automatisch einen Folgejob
ein (via SLURM-Dependency), bis alle Kommentare gelabelt sind.

### Aufruf (einmal starten)

```bash
cd $(ws_find llm_run)/code
sbatch run_labeling_chain.sh conversation_threads_flat.ndjson 48
```

Danach läuft die Kette selbstständig: Job 1 labelt so viel wie in 8h geht,
reiht Job 2 ein (der erst nach Job 1 startet), usw. – bis fertig.

### Wie die Kette stoppt

Der Job vergleicht die Anzahl Ergebniszeilen mit `Kommentare × 7 Tasks`:
- **Alle Zeilen da** → Kette endet mit "KETTE FERTIG"
- **Fortschritt, aber nicht fertig** → nächster Job wird eingereiht
- **Kein Fortschritt** (ein Job schafft nichts Neues) → Kette bricht ab
  (Schutz vor Endlosschleife bei echten Fehlern)

### Kette beobachten und steuern

```bash
squeue -u $USER                      # zeigt laufenden + wartenden Job
tail -f chain_*.out                  # Fortschritt des aktuellen Jobs
scancel <JOBID>                      # Kette stoppen (wartenden Job canceln)
```

### Wichtige Annahme

Die Vollständigkeits-Prüfung nimmt an, dass jeder Kommentar 7 Ergebniszeilen
erzeugt (7 Tasks). Falls ein Kommentar dauerhaft bei einem Task fehlschlägt
(immer `error`), erreicht er nie 7 Zeilen – dann stoppt die Kette beim
Fortschritts-Check mit einer Meldung. Das ist gewollt: lieber stoppen und
das `server_*.log` prüfen als GPU-Zeit endlos verbrennen. Passt du die
Task-Zahl an (`--tasks` mit weniger Tasks), musst du `TASKS_PER_COMMENT`
oben im Chain-Script entsprechend ändern.

### Verhältnis zu den anderen Job-Scripts

| Script                    | Wann                                          |
|---------------------------|-----------------------------------------------|
| `run_labeling.sh`         | Seriell, zum Debuggen, kleine Daten           |
| `run_labeling_async.sh`   | Parallel, EIN Job, passt in 8h                |
| `run_labeling_chain.sh`   | Parallel, MEHRERE Jobs automatisch, große Daten |

---

## Wiederaufnahme nach Abbruch

Nahtlos eingebaut. Der Output-Name ist fest pro Input-Datei
(`labels_<inputname>.ndjson`, kein Zeitstempel). Die Pipeline schreibt
fortlaufend und überspringt bereits gelabelte (comment_id, task)-Paare
automatisch.

**Ablauf bei Wiederaufnahme (z.B. nach 8h-Limit oder Absturz):**

```bash
# 1. Teil-Ergebnisse nach LSDF sichern (Login-Node) - falls noch nicht geschehen
bash $(ws_find llm_run)/code/sync_results_to_lsdf.sh

# 2. Workspace aktualisieren - holt das Teil-Ergebnis aus LSDF zurück (Login-Node)
bash ~/prepare_workspace.sh

# 3. Denselben Job nochmal starten - macht automatisch dort weiter
cd $(ws_find llm_run)/code
sbatch run_labeling.sh conversation_threads_flat.ndjson
```

Der Job meldet im Log ob er fortsetzt (`[2b] WIEDERAUFNAHME: ...`) oder neu
startet (`[2b] NEUSTART: ...`), inklusive Anzahl bereits gelabelter Zeilen.

**Wichtig:** Immer dieselbe Input-Datei mit demselben Namen verwenden, damit
der Output-Name gleich bleibt. Andernfalls beginnt ein frischer Durchlauf.

---

## Tasks

7 Labeling-Tasks + Argument-Extraktion pro Kommentar:
stance_intensity, epistemic_modality, justification_density, responsiveness,
agreement, civility, sarcasm.

Prompts und Schema liegen in `prompts.py`.

---

## Kosten-Hinweis

Jeder Job belegt 4 GPUs für die gesamte Laufzeit (max. 8h auf `accelerated`).
Bei 400 Kommentaren × 8 Calls dauert das Labeling grob 1-2h, der Rest der
GPU-Zeit wäre Verschwendung. Tipp: Job-Zeitlimit an die tatsächliche
Datenmenge anpassen (`#SBATCH --time=`), damit keine GPU-Stunden verfallen.

---

*Fragen: unoim (Arthur)*
