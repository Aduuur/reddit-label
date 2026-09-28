"""
download_model.py  —  Modell aus Hugging Face nach LSDF laden
=============================================================
Lädt ein Modell robust (fortsetzbar bei Abbruch) in die LSDF-Ordnerstruktur,
sodass prepare_workspace.sh es später in den Workspace kopieren kann.

Zielstruktur (passt zur bestehenden Pipeline):
    /lsdf/kit/itz/projects/delib_lab/hf_cache/models/<org>/<name>/

Aufruf (auf dem LOGIN-NODE, im Container, da huggingface_hub dort liegt):
    singularity exec $(ws_find llm_run)/vllm.sif \\
      python3 download_model.py --model Qwen/Qwen2.5-72B-Instruct

Optionen:
    --model      Hugging-Face-Modell-ID (z.B. Qwen/Qwen2.5-72B-Instruct)
    --dest       Zielwurzel (Default: LSDF delib_lab/hf_cache/models)
    --token      HF-Token (sonst aus Umgebung HF_TOKEN / vorheriges hf login)

Wichtig:
  - snapshot_download setzt abgebrochene Downloads fort: bei ~145 GB einfach
    erneut starten, es lädt nur das Fehlende nach.
  - Läuft auf dem LOGIN-NODE (hat Netzzugang zu HF und sieht LSDF).
  - Qwen2.5 ist NICHT gated -> meist kein Token nötig. Llama wäre gated.
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF-Modell-ID, z.B. Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--dest",
                    default="/lsdf/kit/itz/projects/delib_lab/hf_cache/models",
                    help="Zielwurzel für die Modelle")
    ap.add_argument("--token", default=None, help="HF-Token (optional)")
    ap.add_argument("--allow-patterns", default=None,
                    help="Optional: nur bestimmte Dateien, z.B. '*.safetensors'")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("FEHLER: huggingface_hub nicht gefunden. Im Container ausführen:",
              flush=True)
        print("  singularity exec $(ws_find llm_run)/vllm.sif python3 download_model.py ...",
              flush=True)
        sys.exit(1)

    # Zielpfad: <dest>/<org>/<name>
    org_name = args.model  # z.B. "Qwen/Qwen2.5-72B-Instruct"
    target = Path(args.dest) / org_name
    target.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 60, flush=True)
    print("MODELL-DOWNLOAD", flush=True)
    print(f"  Modell: {args.model}", flush=True)
    print(f"  Ziel:   {target}", flush=True)
    print(f"  Token:  {'gesetzt' if token else 'keiner (ok für offene Modelle)'}", flush=True)
    print("=" * 60, flush=True)

    kwargs = dict(
        repo_id=args.model,
        local_dir=str(target),
        # Symlinks vermeiden, damit rsync nach Workspace echte Dateien kopiert
        local_dir_use_symlinks=False,
        resume_download=True,          # <- fortsetzbar
        max_workers=8,
    )
    if token:
        kwargs["token"] = token
    if args.allow_patterns:
        kwargs["allow_patterns"] = [p.strip() for p in args.allow_patterns.split(",")]

    try:
        path = snapshot_download(**kwargs)
    except Exception as e:
        print(f"\nFEHLER beim Download: {e}", flush=True)
        print("Bei Abbruch (Netzwerk/Zeit) einfach denselben Befehl erneut starten -",
              flush=True)
        print("snapshot_download setzt fort und lädt nur das Fehlende.", flush=True)
        sys.exit(1)

    # Kurze Vollständigkeitsanzeige
    safetensors = list(target.glob("*.safetensors"))
    total_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
    print("=" * 60, flush=True)
    print("FERTIG.", flush=True)
    print(f"  Pfad:            {path}", flush=True)
    print(f"  safetensors:     {len(safetensors)} Dateien", flush=True)
    print(f"  Gesamtgröße:     {total_gb:.1f} GB", flush=True)
    print("=" * 60, flush=True)
    print("Nächste Schritte:", flush=True)
    print("  1. Gruppenrechte setzen:  chmod -R g+rX " + str(target), flush=True)
    print("  2. In prepare_workspace.sh MODEL_REL auf den neuen Pfad zeigen lassen", flush=True)
    print("     (oder eine zweite Modell-Variable ergänzen).", flush=True)


if __name__ == "__main__":
    main()
