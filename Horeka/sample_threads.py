"""
sample_threads.py
=================
Wählt VOLLSTÄNDIGE Threads aus einer NDJSON-Datei aus, bis ungefähr eine
Ziel-Anzahl Kommentare erreicht ist. Ganze Threads bleiben zusammen, damit
zu jedem Reply auch der Eltern-Kommentar im Sample ist (wichtig für
responsiveness/agreement - sonst NO_PARENT).

Standardmäßig werden Threads mit mehr Verschachtelung (mehr Antworten auf
Kommentare) bevorzugt, weil die für die kontextabhängigen Dimensionen
interessanter sind als flache Threads, in denen alle Kommentare direkt am
Post hängen.

Aufruf:
    python3 sample_threads.py \
        --input  conversation_threads_resolved.ndjson \
        --output test_threads_500.ndjson \
        --target 500

Optionen:
    --thread-col thread_id     Feld zum Gruppieren
    --min-depth-frac 0.3       nur Threads nehmen, bei denen mind. dieser
                               Anteil der Kommentare depth>0 hat (Reply auf
                               Kommentar). 0 = keine Bedingung.
    --drop-bots                AutoModerator-Kommentare rauswerfen
    --seed 42                  Zufalls-Seed für die Thread-Reihenfolge
"""

from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from collections import defaultdict


def load(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--thread-col", default="thread_id")
    ap.add_argument("--min-depth-frac", type=float, default=0.3)
    ap.add_argument("--drop-bots", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load(args.input)
    if not rows:
        print("Keine Daten geladen.", flush=True); sys.exit(1)
    print(f"Geladen: {len(rows)} Kommentare", flush=True)

    # Optional Bots rauswerfen
    if args.drop_bots:
        before = len(rows)
        rows = [r for r in rows if str(r.get("user","")).lower() not in
                {"automoderator", "[deleted]"}]
        print(f"Bots/deleted entfernt: {before-len(rows)}", flush=True)

    # Nach Thread gruppieren
    threads = defaultdict(list)
    for r in rows:
        threads[r.get(args.thread_col)].append(r)
    print(f"Threads gesamt: {len(threads)}", flush=True)

    # Kennzahl pro Thread: Anteil Kommentare mit depth>0 (echte Replies)
    def depth_frac(comments):
        if not comments:
            return 0.0
        deep = sum(1 for c in comments if (c.get("depth") or 0) > 0)
        return deep / len(comments)

    # Threads filtern nach Mindest-Verschachtelung
    candidates = []
    for tid, comments in threads.items():
        if tid is None:
            continue
        if depth_frac(comments) >= args.min_depth_frac:
            candidates.append((tid, comments))
    print(f"Threads mit depth-Anteil >= {args.min_depth_frac}: {len(candidates)}", flush=True)

    if not candidates:
        print("Keine Threads erfüllen das depth-Kriterium. Senke --min-depth-frac.", flush=True)
        sys.exit(1)

    # Threads mischen (reproduzierbar) und bis zum Ziel aufsammeln
    random.seed(args.seed)
    random.shuffle(candidates)

    selected = []
    count = 0
    used_threads = 0
    for tid, comments in candidates:
        if count >= args.target:
            break
        # Kommentare eines Threads in stabiler Reihenfolge (nach created_utc)
        comments_sorted = sorted(comments, key=lambda c: c.get("created_utc") or 0)
        selected.extend(comments_sorted)
        count += len(comments_sorted)
        used_threads += 1

    # Schreiben
    with open(args.output, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistik
    n_with_pred = sum(1 for r in selected if r.get("predecessor"))
    n_toplevel = len(selected) - n_with_pred
    print("=" * 55, flush=True)
    print(f"Ausgewählt: {len(selected)} Kommentare aus {used_threads} Threads", flush=True)
    print(f"  mit Eltern-Text (predecessor):  {n_with_pred}", flush=True)
    print(f"  Top-Level (kein predecessor):   {n_toplevel}", flush=True)
    print(f"Geschrieben nach: {args.output}", flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    main()
