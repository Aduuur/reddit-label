"""
resolve_predecessor.py
======================
Löst das Feld `predecessor` (Text des Eltern-Kommentars) über die `parent_id`
auf. Reddit-Threads enthalten pro Zeile eine comment_id und eine parent_id;
der Text des Elternkommentars steht in einer ANDEREN Zeile mit passender
comment_id. Dieses Skript verknüpft beides.

Ablauf:
  1. Erster Durchgang: Nachschlagetabelle {comment_id -> body} aufbauen.
  2. Zweiter Durchgang: pro Zeile parent_id in der Tabelle suchen; wenn
     gefunden, dessen body als `predecessor` eintragen.

Sonderfälle (bewusst behandelt):
  - parent_id == link_id: Der Kommentar hängt direkt am Post (Submission),
    nicht an einem Kommentar. Der Post-Text liegt in dieser Datei i.d.R.
    nicht als Zeile vor -> predecessor bleibt leer (None). Die Labeling-
    Pipeline gibt dann für responsiveness/agreement korrekt ABSTAIN zurück.
  - parent_id nicht in der Tabelle (z.B. Elternkommentar gelöscht oder nicht
    im Sample): predecessor bleibt leer.
  - Ein bereits vorhandenes, nicht-leeres predecessor-Feld wird NICHT
    überschrieben (falls die Datei schon teilweise aufgelöst war).

Aufruf:
    python3 resolve_predecessor.py \
        --input  conversation_threads_flat.ndjson \
        --output conversation_threads_resolved.ndjson

Optional:
    --id-col comment_id  --parent-col parent_id  --link-col link_id
    --text-col body      --pred-col predecessor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_lookup(path: Path, id_col: str, text_col: str):
    """Erster Durchgang: comment_id -> body. Überspringt kaputte Zeilen."""
    lookup = {}
    total = skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            cid = obj.get(id_col)
            body = obj.get(text_col)
            if cid is not None and body is not None:
                lookup[str(cid)] = body
    return lookup, total, skipped


def resolve(path_in: Path, path_out: Path, lookup: dict,
            id_col: str, parent_col: str, link_col: str,
            text_col: str, pred_col: str):
    """Zweiter Durchgang: predecessor eintragen und neue Datei schreiben."""
    stats = {
        "lines": 0, "skipped_json": 0,
        "resolved_from_comment": 0,   # Parent war ein Kommentar (aufgelöst)
        "toplevel_post": 0,           # Parent == Post (kein Kommentar-Text)
        "parent_missing": 0,          # parent_id nicht in der Tabelle
        "already_had_pred": 0,        # predecessor war schon gesetzt
        "no_parent_field": 0,         # gar keine parent_id vorhanden
    }
    with open(path_in, "r", encoding="utf-8") as fin, \
         open(path_out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_json"] += 1
                continue
            stats["lines"] += 1

            # Vorhandenes, nicht-leeres predecessor nicht überschreiben
            existing = obj.get(pred_col)
            if existing is not None and str(existing).strip():
                stats["already_had_pred"] += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            parent = obj.get(parent_col)
            link = obj.get(link_col)

            if parent is None:
                obj[pred_col] = None
                stats["no_parent_field"] += 1
            else:
                parent = str(parent)
                # Top-Level? parent zeigt auf den Post (== link_id)
                if link is not None and parent == str(link):
                    # Falls der Post ausnahmsweise doch als Zeile existiert,
                    # nehmen wir seinen Text; sonst None.
                    if parent in lookup:
                        obj[pred_col] = lookup[parent]
                        stats["resolved_from_comment"] += 1
                    else:
                        obj[pred_col] = None
                        stats["toplevel_post"] += 1
                else:
                    # Parent ist ein Kommentar -> Text nachschlagen
                    if parent in lookup:
                        obj[pred_col] = lookup[parent]
                        stats["resolved_from_comment"] += 1
                    else:
                        obj[pred_col] = None
                        stats["parent_missing"] += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser(description="predecessor über parent_id auflösen")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--id-col", default="comment_id")
    ap.add_argument("--parent-col", default="parent_id")
    ap.add_argument("--link-col", default="link_id")
    ap.add_argument("--text-col", default="body")
    ap.add_argument("--pred-col", default="predecessor")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"FEHLER: Eingabedatei nicht gefunden: {args.input}", flush=True)
        sys.exit(1)

    print("=" * 60, flush=True)
    print("PREDECESSOR-AUFLÖSUNG", flush=True)
    print(f"  Input:  {args.input}", flush=True)
    print(f"  Output: {args.output}", flush=True)
    print("=" * 60, flush=True)

    print("[1/2] Baue Nachschlagetabelle (comment_id -> body) ...", flush=True)
    lookup, total, skipped = build_lookup(args.input, args.id_col, args.text_col)
    print(f"      {len(lookup)} Kommentare indexiert "
          f"({total} Zeilen gelesen, {skipped} kaputte übersprungen).", flush=True)

    print("[2/2] Löse predecessor auf und schreibe Ausgabe ...", flush=True)
    stats = resolve(args.input, args.output, lookup,
                    args.id_col, args.parent_col, args.link_col,
                    args.text_col, args.pred_col)

    print("=" * 60, flush=True)
    print("FERTIG. Statistik:", flush=True)
    print(f"  Zeilen geschrieben:            {stats['lines']}", flush=True)
    print(f"  predecessor aufgelöst:         {stats['resolved_from_comment']}", flush=True)
    print(f"  Top-Level (Parent = Post):     {stats['toplevel_post']}  -> predecessor leer", flush=True)
    print(f"  Parent-ID nicht gefunden:      {stats['parent_missing']}  -> predecessor leer", flush=True)
    print(f"  hatte schon predecessor:       {stats['already_had_pred']}", flush=True)
    print(f"  ohne parent_id-Feld:           {stats['no_parent_field']}", flush=True)
    if stats["skipped_json"]:
        print(f"  kaputte JSON-Zeilen:           {stats['skipped_json']}", flush=True)
    resolved = stats["resolved_from_comment"]
    if stats["lines"]:
        pct = 100.0 * resolved / stats["lines"]
        print(f"\n  => {resolved} von {stats['lines']} Kommentaren ({pct:.1f}%) "
              f"haben jetzt einen Eltern-Text.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
