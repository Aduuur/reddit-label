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
from typing import Optional


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


def build_post_body(obj: dict, title_col: str, selftext_col: str) -> Optional[str]:
    """Setzt den Post-Text aus Titel + Selftext zusammen (für Top-Level-
    Kommentare, deren Vorgänger der Post ist)."""
    title = (obj.get(title_col) or "").strip()
    selftext = (obj.get(selftext_col) or "").strip()
    if title and selftext:
        return f"{title}\n\n{selftext}"
    return (title or selftext) or None


def resolve(path_in: Path, path_out: Path, lookup: dict,
            id_col: str, parent_col: str, link_col: str,
            text_col: str, pred_col: str,
            title_col: str = "submission_title",
            selftext_col: str = "submission_selftext"):
    """Zweiter Durchgang: predecessor eintragen und neue Datei schreiben."""
    stats = {
        "lines": 0, "skipped_json": 0,
        "resolved_from_comment": 0,   # Parent war ein Kommentar (aufgelöst)
        "resolved_from_post": 0,      # Top-Level -> Post-Text als predecessor
        "is_post": 0,                 # Post-Zeile selbst (kein predecessor)
        "toplevel_no_text": 0,        # Top-Level, aber kein Post-Text vorhanden
        "parent_missing": 0,          # parent_id nicht in der Tabelle
        "already_had_pred": 0,        # predecessor war schon gesetzt
        "no_parent_field": 0,         # gar keine parent_id vorhanden (= Post)
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

            # Post-Zeile selbst (depth == -1 oder is_post): kein predecessor
            if obj.get("is_post") is True or obj.get("depth") == -1:
                obj[pred_col] = None
                stats["is_post"] += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

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
                    if parent in lookup:
                        # Post existiert als Zeile -> dessen Body nehmen
                        obj[pred_col] = lookup[parent]
                        stats["resolved_from_comment"] += 1
                    else:
                        # Fallback: Post-Text aus Titel + Selftext zusammensetzen
                        post_body = build_post_body(obj, title_col, selftext_col)
                        if post_body:
                            obj[pred_col] = post_body
                            stats["resolved_from_post"] += 1
                        else:
                            obj[pred_col] = None
                            stats["toplevel_no_text"] += 1
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
    print(f"  predecessor aus Kommentar:     {stats['resolved_from_comment']}", flush=True)
    print(f"  predecessor aus Post-Text:     {stats['resolved_from_post']}  (Top-Level -> Post)", flush=True)
    print(f"  Post-Zeilen (kein predecessor):{stats['is_post']}", flush=True)
    print(f"  Top-Level ohne Post-Text:      {stats['toplevel_no_text']}  -> predecessor leer", flush=True)
    print(f"  Parent-ID nicht gefunden:      {stats['parent_missing']}  -> predecessor leer", flush=True)
    print(f"  hatte schon predecessor:       {stats['already_had_pred']}", flush=True)
    print(f"  ohne parent_id-Feld:           {stats['no_parent_field']}", flush=True)
    if stats["skipped_json"]:
        print(f"  kaputte JSON-Zeilen:           {stats['skipped_json']}", flush=True)
    resolved = stats["resolved_from_comment"] + stats["resolved_from_post"]
    # Kommentare = alle Zeilen minus die Post-Zeilen
    n_comments = stats["lines"] - stats["is_post"]
    if n_comments:
        pct = 100.0 * resolved / n_comments
        print(f"\n  => {resolved} von {n_comments} Kommentaren ({pct:.1f}%) "
              f"haben jetzt einen Eltern-Text (Kommentar oder Post).", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
