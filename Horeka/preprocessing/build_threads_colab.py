# =============================================================================
# Reddit-Threads vollständig zusammenfügen  (Colab-ready)
# -----------------------------------------------------------------------------
# Baut aus Submissions + Kommentaren eine flache NDJSON-Datei (ein Datensatz
# pro Kommentar) mit berechneter depth und root_id.
#
# Robust gegen:
#   - fehlende Zwischen-Kommentare (Waisen werden NICHT abgeschnitten)
#   - doppelte IDs (werden mit Warnung entfernt, nicht still verschluckt)
#   - Reddit-Präfixe (t1_, t3_)
#   - Zyklen in der Elternkette (Schutz eingebaut)
#
# Am Ende steht eine Vollständigkeits-Prüfung: kein Kommentar aus dem Sample
# darf im Output fehlen.
# =============================================================================

from google.colab import drive
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List
import json

import pandas as pd
import numpy as np

drive.mount("/content/drive")

# -----------------------------------------------------------------------------
# KONFIGURATION  – hier ggf. Pfade anpassen
# -----------------------------------------------------------------------------
DATA_DIR      = Path("/content/drive/MyDrive/reddit_data")
SUBMISSIONS   = DATA_DIR / "sampled.ndjson"          # gesampelte Submissions
COMMENT_GLOB  = "*_comments.jsonl"                    # alle Kommentar-Dumps
OUT_PATH      = DATA_DIR / "conversation_threads_flat.ndjson"
MIN_COMMENTS  = 1        # Submissions mit weniger Kommentaren rauswerfen (0 = alle behalten)
INCLUDE_PATH  = False    # True: ancestor_path (Liste der Vorfahren) mitschreiben


# =============================================================================
# 1) Helfer
# =============================================================================
def strip_prefix(x) -> Optional[str]:
    """Entfernt Reddit-Präfixe wie t1_ / t3_. Gibt None bei fehlendem Wert."""
    if pd.isna(x):
        return None
    x = str(x)
    return x.split("_", 1)[1] if "_" in x else x


def read_jsonl_safe(path: Path, max_preview: int = 3) -> Tuple[pd.DataFrame, int]:
    """Liest JSONL zeilenweise, überspringt kaputte Zeilen, meldet deren Anzahl."""
    rows, skipped, first_errors = [], 0, []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                skipped += 1
                if len(first_errors) < max_preview:
                    first_errors.append((i, str(e)))
    if skipped:
        print(f"[WARN] {path.name}: {skipped} kaputte Zeilen übersprungen")
        for ln, msg in first_errors:
            print(f"        - Zeile {ln}: {msg}")
    return pd.DataFrame(rows), skipped


def to_json_safe(obj):
    """Macht numpy/pandas-Typen JSON-serialisierbar."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


# =============================================================================
# 2) Submissions laden + filtern
# =============================================================================
print("=" * 60)
print("SUBMISSIONS")
print("=" * 60)
assert SUBMISSIONS.exists(), f"Submissions nicht gefunden: {SUBMISSIONS}"

df_submissions = pd.read_json(SUBMISSIONS, lines=True)
print(f"Submissions geladen: {len(df_submissions):,}")

# num_comments als Zahl interpretieren (fehlende -> 0)
df_submissions["num_comments"] = (
    pd.to_numeric(df_submissions.get("num_comments"), errors="coerce")
    .fillna(0).astype("int64")
)

before = len(df_submissions)
df_submissions_filtered = df_submissions.loc[
    df_submissions["num_comments"] >= MIN_COMMENTS
].copy()
after = len(df_submissions_filtered)
removed = before - after
share = (removed / before * 100.0) if before else 0.0
print(f"Filter num_comments >= {MIN_COMMENTS}: {after:,} behalten, "
      f"{removed:,} entfernt ({share:.1f}%)")


# =============================================================================
# 3) Kommentare laden (alle *_comments.jsonl) + zusammenführen
# =============================================================================
print("\n" + "=" * 60)
print("KOMMENTARE")
print("=" * 60)
comment_files = sorted(DATA_DIR.glob(COMMENT_GLOB))
print(f"Gefundene Dateien ({len(comment_files)}):")
for f in comment_files:
    print("  -", f.name)

dfs, total_skipped = [], 0
for p in comment_files:
    df_part, skipped = read_jsonl_safe(p)
    total_skipped += skipped
    if not df_part.empty:
        df_part["source_file"] = p.name
        dfs.append(df_part)

assert dfs, "Keine Kommentar-Dateien geladen!"
df_comments = pd.concat(dfs, ignore_index=True)
print(f"\nKommentare geladen: {len(df_comments):,} "
      f"(kaputte Zeilen gesamt: {total_skipped})")


# =============================================================================
# 4) IDs normalisieren
# =============================================================================
df_submissions_filtered = df_submissions_filtered.copy()
df_submissions_filtered["submission_id"] = df_submissions_filtered["id"].apply(strip_prefix)

df_comments["comment_id"] = df_comments["id"].apply(strip_prefix)
df_comments["parent"]     = df_comments["parent_id"].apply(strip_prefix)
df_comments["link"]       = df_comments["link_id"].apply(strip_prefix)


# =============================================================================
# 5) Duplikate entfernen (mit Warnung, statt sie still zu verschlucken)
# =============================================================================
dup_sub = df_submissions_filtered["submission_id"].duplicated().sum()
if dup_sub:
    print(f"[WARN] {dup_sub} doppelte submission_id entfernt")
    df_submissions_filtered = df_submissions_filtered.drop_duplicates(
        subset="submission_id", keep="first").copy()

dup_com = df_comments["comment_id"].duplicated().sum()
if dup_com:
    print(f"[WARN] {dup_com} doppelte comment_id entfernt")
    df_comments = df_comments.drop_duplicates(
        subset="comment_id", keep="first").copy()


# =============================================================================
# 6) Kommentare auf gesampelte Submissions einschränken
# =============================================================================
sampled_submission_ids = set(df_submissions_filtered["submission_id"].dropna().astype(str))
print(f"\nGesampelte Submissions: {len(sampled_submission_ids):,}")

df_comments_in_sample = df_comments[
    df_comments["link"].isin(sampled_submission_ids)
].copy()
print(f"Kommentare in gesampelten Submissions: {len(df_comments_in_sample):,}")


# =============================================================================
# 7) Lookups aufbauen
# =============================================================================
comment_by_id: Dict[str, Dict[str, Any]] = (
    df_comments_in_sample.set_index("comment_id").to_dict(orient="index")
)
comment_id_set = set(comment_by_id.keys())

submissions_lookup: Dict[str, Dict[str, Any]] = (
    df_submissions_filtered.set_index("submission_id").to_dict(orient="index")
)


# =============================================================================
# 8) depth + root_id je Kommentar über AUFSTIEG der Elternkette
#    Kernidee: statt vom Post nach unten (was Waisen abschneidet) laufen wir
#    von JEDEM Kommentar nach oben. So geht garantiert keiner verloren.
#    Ein Kommentar ist Wurzel (depth 0), wenn sein Parent die Submission ist
#    ODER nicht im Sample liegt (Waise). Ergebnisse werden memoisiert.
# =============================================================================
depth_cache: Dict[str, int] = {}
root_cache: Dict[str, str] = {}

def resolve_up(cid: str) -> Tuple[int, str]:
    chain: List[str] = []
    seen = set()
    cur = cid
    while True:
        # Bereits berechnet? -> Kette dranhängen
        if cur in depth_cache:
            base_d, base_r = depth_cache[cur], root_cache[cur]
            for i, c in enumerate(reversed(chain)):
                depth_cache[c] = base_d + i + 1
                root_cache[c] = base_r
            return depth_cache[cid], root_cache[cid]

        # Zyklus? -> defensiv abbrechen, cur als Wurzel
        if cur in seen:
            depth_cache[cur] = 0
            root_cache[cur] = cur
            for i, c in enumerate(reversed(chain)):
                depth_cache[c] = i + 1
                root_cache[c] = cur
            return depth_cache[cid], root_cache[cid]
        seen.add(cur)

        c = comment_by_id.get(cur)
        parent = c.get("parent") if c else None

        # Wurzelbedingung: Parent = Submission ODER Parent nicht im Sample (Waise)
        if parent is None or parent in sampled_submission_ids or parent not in comment_id_set:
            depth_cache[cur] = 0
            root_cache[cur] = cur
            for i, c2 in enumerate(reversed(chain)):
                depth_cache[c2] = i + 1
                root_cache[c2] = cur
            return depth_cache[cid], root_cache[cid]

        chain.append(cur)
        cur = parent


# =============================================================================
# 9) Flache Datensätze bauen (jeder Kommentar im Sample, angereichert)
# =============================================================================
all_rows: List[Dict[str, Any]] = []

for cid in comment_by_id:
    depth, root = resolve_up(cid)
    c = comment_by_id[cid]
    sid = c.get("link")
    srow = submissions_lookup.get(sid, {})

    rec = {
        "thread_id": sid,
        "submission_id": sid,
        "root_id": root,
        "comment_id": cid,
        "parent_id": c.get("parent"),
        "link_id": c.get("link"),
        "body": c.get("body"),
        "user": c.get("author"),
        "score": c.get("score"),
        "created_utc": c.get("created_utc"),
        "permalink": c.get("permalink"),
        "subreddit": c.get("subreddit"),
        "depth": int(depth),
        # Submission-Metadaten denormalisiert dazu
        "submission_title": srow.get("title"),
        "submission_selftext": srow.get("selftext"),
        "submission_author": srow.get("author"),
        "submission_score": srow.get("score"),
        "submission_created_utc": srow.get("created_utc"),
        "submission_subreddit": srow.get("subreddit"),
        "submission_url": srow.get("url"),
        "best_topic_index": srow.get("best_topic_index"),
        "best_topic_similarity": srow.get("best_topic_similarity"),
    }
    if INCLUDE_PATH:
        # Vorfahrenkette rekonstruieren (root -> ... -> parent)
        path, cur = [], c.get("parent")
        guard = 0
        while cur in comment_id_set and guard < 10000:
            path.append(cur)
            cur = comment_by_id[cur].get("parent")
            guard += 1
        rec["ancestor_path"] = json.dumps(list(reversed(path)), ensure_ascii=False)

    all_rows.append(rec)

# --- Post selbst als Wurzelknoten pro Submission hinzufügen ---
# Der Post erscheint als eigene Zeile (comment_id == submission_id, depth == -1).
# body = Titel + Selftext. Dadurch haben Top-Level-Kommentare beim Auflösen
# einen echten Vorgänger (den Post). Nur der Post selbst hat keinen predecessor.
def build_post_body(srow: Dict[str, Any]) -> str:
    title = (srow.get("title") or "").strip()
    selftext = (srow.get("selftext") or "").strip()
    if title and selftext:
        return f"{title}\n\n{selftext}"
    return title or selftext

# Nur Submissions, die auch tatsächlich Kommentare im Sample haben, brauchen
# einen Wurzelknoten (sonst blähen leere Posts den Datensatz auf).
subs_with_comments = {c.get("link") for c in comment_by_id.values()}

n_posts = 0
for sid in subs_with_comments:
    if sid is None:
        continue
    srow = submissions_lookup.get(sid, {})
    post_rec = {
        "thread_id": sid,
        "submission_id": sid,
        "root_id": sid,            # der Post ist seine eigene Wurzel
        "comment_id": sid,         # comment_id == submission_id
        "parent_id": None,         # der Post hat KEINEN Vorgänger
        "link_id": sid,
        "body": build_post_body(srow),
        "user": srow.get("author"),
        "score": srow.get("score"),
        "created_utc": srow.get("created_utc"),
        "permalink": None,
        "subreddit": srow.get("subreddit"),
        "depth": -1,               # Post-Ebene (Kommentare beginnen bei 0)
        "is_post": True,           # Marker, um Posts leicht herausfiltern zu können
        "submission_title": srow.get("title"),
        "submission_selftext": srow.get("selftext"),
        "submission_author": srow.get("author"),
        "submission_score": srow.get("score"),
        "submission_created_utc": srow.get("created_utc"),
        "submission_subreddit": srow.get("subreddit"),
        "submission_url": srow.get("url"),
        "best_topic_index": srow.get("best_topic_index"),
        "best_topic_similarity": srow.get("best_topic_similarity"),
    }
    if INCLUDE_PATH:
        post_rec["ancestor_path"] = json.dumps([], ensure_ascii=False)
    all_rows.append(post_rec)
    n_posts += 1
print(f"Post-Wurzelknoten hinzugefügt: {n_posts:,}")

df_threads_flat = pd.DataFrame(all_rows)
print(f"\nFlache Datensätze gebaut: {len(df_threads_flat):,}")


# =============================================================================
# 10) VOLLSTÄNDIGKEITS-PRÜFUNG  (das Wichtigste)
# =============================================================================
print("\n" + "=" * 60)
print("VOLLSTÄNDIGKEITS-PRÜFUNG")
print("=" * 60)

# Kommentare und Post-Zeilen getrennt betrachten
is_post = df_threads_flat["depth"] == -1
df_comments_only = df_threads_flat[~is_post]
n_posts_out = int(is_post.sum())

reachable = set(df_comments_only["comment_id"].astype(str))
all_in_sample = set(df_comments_in_sample["comment_id"].astype(str))
lost = all_in_sample - reachable

print(f"Kommentare im Sample:  {len(all_in_sample):,}")
print(f"Kommentare im Output:  {len(reachable):,}")
print(f"Post-Wurzelknoten:     {n_posts_out:,}")
print(f"Zeilen gesamt:         {len(df_threads_flat):,}")
print(f"Verloren:              {len(lost):,}")

# Struktur-Kennzahlen (nur über Kommentare, nicht über Posts)
n_toplevel = int((df_comments_only["depth"] == 0).mean() * 100)
orphan_roots = sum(
    1 for cid in comment_by_id
    if (comment_by_id[cid].get("parent") not in sampled_submission_ids
        and comment_by_id[cid].get("parent") not in comment_id_set
        and comment_by_id[cid].get("parent") is not None)
)
print(f"Top-Level-Anteil:      {n_toplevel}%")
print(f"Waisen-Wurzeln (Parent fehlt im Sample): {orphan_roots:,}")

assert len(df_comments_only) == len(df_comments_in_sample), (
    "FEHLER: Kommentar-Zeilenzahl weicht von Sample ab – es gingen Kommentare verloren!"
)
assert len(lost) == 0, f"FEHLER: {len(lost)} Kommentare fehlen im Output!"
print("\n✓ Vollständig: jeder Kommentar aus dem Sample ist im Output (+ Post-Wurzelknoten).")


# =============================================================================
# 11) Als NDJSON schreiben
# =============================================================================
with OUT_PATH.open("w", encoding="utf-8") as f:
    for rec in df_threads_flat.to_dict(orient="records"):
        f.write(json.dumps(to_json_safe(rec), ensure_ascii=False) + "\n")

print(f"\nGeschrieben nach: {OUT_PATH.resolve()}")
print(f"Zeilen: {len(df_threads_flat):,}")
df_threads_flat.head()
