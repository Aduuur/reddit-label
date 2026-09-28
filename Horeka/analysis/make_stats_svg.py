"""
make_stats_svg.py
=================
Wie make_stats.py, aber OHNE matplotlib - erzeugt SVG-Diagramme mit reinem
Python (Standardbibliothek). Läuft überall, auch im Container ohne Extra-Pakete.

SVG-Dateien öffnest du im Browser (Doppelklick) und kannst sie direkt in
Folien/Docs einfügen oder als PNG exportieren (im Browser: Rechtsklick).

Aufruf (auf HoreKa, KEIN Container nötig - pure Python):
    python3 make_stats_svg.py \
        --input  $(ws_find llm_run)/results/labels_test_300_resolved.ndjson \
        --outdir $(ws_find llm_run)/results/stats
"""

from __future__ import annotations
import argparse, json, sys, html
from pathlib import Path
from collections import defaultdict, Counter

BLUE="#2E5496"; GREEN="#548235"; GRAY="#9AA0A6"; RED="#A61C00"; TEAL="#2A9D8F"; DARK="#333333"; ORANGE="#BF8F00"

ALL_TASKS = ["stance_intensity","epistemic_modality","justification_density",
             "responsiveness","agreement","civility","sarcasm"]
NUMERIC = {
    "stance_intensity": ("Stance Intensity (1=dagegen … 6=dafür)", list(range(1,7))),
    "civility":         ("Civility (1=unhöflich … 6=höflich)",     list(range(1,7))),
    "sarcasm":          ("Sarcasm (0=nein, 1=ja)",                 [0,1]),
}

def esc(s): return html.escape(str(s))

def load(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rows

def val_of(r):
    res=r.get("result",{})
    if "error" in res: return "ERROR"
    return res.get("score", res.get("label"))

def conf_of(r):
    c=r.get("result",{}).get("confidence")
    return c if isinstance(c,(int,float)) else None

def svg_header(w,h,title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w/2}" y="30" font-size="20" font-weight="bold" '
            f'text-anchor="middle" fill="{DARK}">{esc(title)}</text>')

def svg_header_small(w,h,title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w/2}" y="26" font-size="14" font-weight="bold" '
            f'text-anchor="middle" fill="{DARK}">{esc(title)}</text>')

def stacked_bars(path, tasks, labeled, abstained, errored, n_comments):
    W,H=980,60+len(tasks)*54+40
    left,right=200,130
    plot_w=W-left-right
    maxv=max((l+a+e) for l,a,e in zip(labeled,abstained,errored)) or 1
    s=[svg_header(W,H,f"Label-Abdeckung je Dimension (n={n_comments})")]
    y0=60
    for i,t in enumerate(tasks):
        y=y0+i*54
        l,a,e=labeled[i],abstained[i],errored[i]
        lw=plot_w*l/maxv; aw=plot_w*a/maxv; ew=plot_w*e/maxv
        s.append(f'<text x="{left-10}" y="{y+22}" font-size="14" text-anchor="end" fill="{DARK}">{esc(t)}</text>')
        x=left
        s.append(f'<rect x="{x}" y="{y}" width="{lw:.1f}" height="32" fill="{GREEN}"/>'); x+=lw
        s.append(f'<rect x="{x}" y="{y}" width="{aw:.1f}" height="32" fill="{GRAY}"/>'); x+=aw
        if e: s.append(f'<rect x="{x}" y="{y}" width="{ew:.1f}" height="32" fill="{RED}"/>'); x+=ew
        pct=100*l/(l+a+e) if (l+a+e) else 0
        s.append(f'<text x="{x+8}" y="{y+22}" font-size="13" fill="{DARK}">{pct:.0f}% gelabelt</text>')
    # Legende
    ly=H-24
    for lx,(col,lab) in zip([left, left+150, left+300],[(GREEN,"gelabelt"),(GRAY,"ABSTAIN"),(RED,"Fehler")]):
        s.append(f'<rect x="{lx}" y="{ly-11}" width="14" height="14" fill="{col}"/>')
        s.append(f'<text x="{lx+20}" y="{ly}" font-size="13" fill="{DARK}">{lab}</text>')
    s.append("</svg>")
    path.write_text("".join(s),encoding="utf-8")

def histogram(path, task, title, lo, hi, nbins, by_task):
    """Histogramm für kontinuierliche Werte (agreement, responsiveness,
    epistemic_modality, justification_density)."""
    vs=[val_of(r) for r in by_task.get(task,[])]
    nums=[float(v) for v in vs if isinstance(v,(int,float))]
    n_abst=sum(1 for v in vs if v=="ABSTAIN")
    # justification_density ist nach oben offen -> hi dynamisch
    if task=="justification_density" and nums:
        hi=max(hi, max(nums))
    width=(hi-lo)/nbins if nbins else 1
    counts=[0]*nbins
    for v in nums:
        idx=int((v-lo)/width) if width else 0
        if idx>=nbins: idx=nbins-1
        if idx<0: idx=0
        counts[idx]+=1
    W,H=520,340
    left,bottom=50,54; top=64; right=24
    plot_w=W-left-right; plot_h=H-top-bottom
    maxv=max(counts) if counts and max(counts)>0 else 1
    bw=plot_w/nbins
    s=[svg_header_small(W,H,title)]
    s.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#ccc"/>')
    for i in range(nbins):
        h=plot_h*counts[i]/maxv
        x=left+i*bw
        y=top+plot_h-h
        s.append(f'<rect x="{x+1:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{h:.1f}" fill="{BLUE}"/>')
        if counts[i]>0:
            s.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" font-size="10" text-anchor="middle" fill="{DARK}">{counts[i]}</text>')
    # Achsenbeschriftung: lo, Mitte, hi
    for frac,val in [(0,lo),(0.5,(lo+hi)/2),(1,hi)]:
        x=left+plot_w*frac
        s.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" font-size="11" text-anchor="middle" fill="{DARK}">{val:.1f}</text>')
    s.append(f'<text x="{W-right}" y="{top-8}" font-size="11" text-anchor="end" fill="{GRAY}">ABSTAIN: {n_abst}</text>')
    s.append("</svg>")
    path.write_text("".join(s),encoding="utf-8")


def distribution_bars(path, task, title, ticks, by_task):
    vs=[val_of(r) for r in by_task.get(task,[])]
    nums=[v for v in vs if isinstance(v,(int,float))]
    cnt=Counter(int(round(v)) for v in nums)
    n_abst=sum(1 for v in vs if v=="ABSTAIN")
    heights=[cnt.get(t,0) for t in ticks]
    W,H=520,340
    left,bottom=50,50; top=64; right=24
    plot_w=W-left-right; plot_h=H-top-bottom
    maxv=max(heights) if heights and max(heights)>0 else 1
    bw=plot_w/len(ticks)*0.7; gap=plot_w/len(ticks)
    s=[svg_header_small(W,H,title)]
    # Achse
    s.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#ccc"/>')
    for i,t in enumerate(ticks):
        h=plot_h*heights[i]/maxv
        x=left+i*gap+(gap-bw)/2
        y=top+plot_h-h
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{BLUE}"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{top+plot_h+18}" font-size="12" text-anchor="middle" fill="{DARK}">{esc(t)}</text>')
        if heights[i]>0:
            s.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" font-size="11" text-anchor="middle" fill="{DARK}">{heights[i]}</text>')
    s.append(f'<text x="{W-right}" y="{top-8}" font-size="11" text-anchor="end" fill="{GRAY}">ABSTAIN: {n_abst}</text>')
    s.append("</svg>")
    path.write_text("".join(s),encoding="utf-8")

def confidence_bars(path, tasks, by_task):
    means=[]; labs=[]
    for t in tasks:
        cs=[conf_of(r) for r in by_task[t]]; cs=[c for c in cs if c is not None]
        if cs: means.append(sum(cs)/len(cs)); labs.append(t)
    order=sorted(range(len(means)),key=lambda i:means[i])
    means=[means[i] for i in order]; labs=[labs[i] for i in order]
    W,H=760,60+len(labs)*46+30
    left,right=200,80; plot_w=W-left-right
    s=[svg_header(W,H,"Durchschnittliche Modell-Konfidenz je Dimension")]
    y0=60
    for i,(m,lab) in enumerate(zip(means,labs)):
        y=y0+i*46
        w=plot_w*m
        s.append(f'<text x="{left-10}" y="{y+20}" font-size="14" text-anchor="end" fill="{DARK}">{esc(lab)}</text>')
        s.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="28" fill="#eee"/>')
        s.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="28" fill="{TEAL}"/>')
        s.append(f'<text x="{left+w+8:.1f}" y="{y+20}" font-size="13" fill="{DARK}">{m:.2f}</text>')
    s.append("</svg>")
    path.write_text("".join(s),encoding="utf-8")

def n_tokens_of(r):
    v = r.get("n_tokens")
    return v if isinstance(v, (int, float)) else None


def confidence_hist_by_length(path, rows, short_thr=15):
    """Konfidenz-Verteilung in 10 Buckets, getrennt nach kurzen und langen
    Kommentaren. Zeigt, ob das Modell bei kurzen Texten unsicherer ist."""
    short = [0]*10
    long = [0]*10
    for r in rows:
        v = val_of(r)
        if v in ("ABSTAIN", "ERROR"):
            continue
        c = conf_of(r)
        if c is None:
            continue
        b = min(int(c*10), 9)
        nt = n_tokens_of(r)
        if nt is not None and nt < short_thr:
            short[b] += 1
        else:
            long[b] += 1
    ks, ls = sum(short) or 1, sum(long) or 1
    # Als Prozent pro Gruppe (damit vergleichbar trotz unterschiedlicher Menge)
    short_p = [x/ks*100 for x in short]
    long_p = [x/ls*100 for x in long]

    W, H = 820, 420
    left, right, top, bottom = 60, 130, 70, 60
    plot_w = W-left-right
    plot_h = H-top-bottom
    maxv = max(max(short_p), max(long_p), 1)
    nb = 10
    gap = plot_w/nb
    bw = gap*0.38
    s = [svg_header_small(W, H, f"Konfidenz-Verteilung: kurz (<{short_thr} Tokens) vs. lang")]
    s.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#ccc"/>')
    labels = ["0-.1",".1-.2",".2-.3",".3-.4",".4-.5",".5-.6",".6-.7",".7-.8",".8-.9",".9-1"]
    for i in range(nb):
        x0 = left+i*gap+(gap-2*bw)/2
        hs = plot_h*short_p[i]/maxv
        hl = plot_h*long_p[i]/maxv
        s.append(f'<rect x="{x0:.1f}" y="{top+plot_h-hs:.1f}" width="{bw:.1f}" height="{hs:.1f}" fill="{ORANGE}"/>')
        s.append(f'<rect x="{x0+bw:.1f}" y="{top+plot_h-hl:.1f}" width="{bw:.1f}" height="{hl:.1f}" fill="{BLUE}"/>')
        s.append(f'<text x="{x0+bw:.1f}" y="{top+plot_h+16}" font-size="9" text-anchor="middle" fill="{DARK}">{labels[i]}</text>')
    # Legende
    ly = top+6
    s.append(f'<rect x="{left+plot_w+14}" y="{ly}" width="12" height="12" fill="{ORANGE}"/>')
    s.append(f'<text x="{left+plot_w+30}" y="{ly+11}" font-size="12" fill="{DARK}">kurz</text>')
    s.append(f'<rect x="{left+plot_w+14}" y="{ly+22}" width="12" height="12" fill="{BLUE}"/>')
    s.append(f'<text x="{left+plot_w+30}" y="{ly+33}" font-size="12" fill="{DARK}">lang</text>')
    s.append(f'<text x="{left-6}" y="{top-14}" font-size="11" fill="{GRAY}">% je Gruppe</text>')
    s.append("</svg>")
    path.write_text("".join(s), encoding="utf-8")


def confidence_hist_per_task(path, tasks, by_task):
    """Pro Task: Anteil der Labels in drei Konfidenz-Bändern (niedrig/mittel/hoch).
    Zeigt, welche Dimensionen overconfident sind."""
    LOW, MID = 0.5, 0.7
    rows_data = []
    for t in tasks:
        cs = [conf_of(r) for r in by_task[t] if val_of(r) not in ("ABSTAIN","ERROR")]
        cs = [c for c in cs if c is not None]
        if not cs:
            continue
        n = len(cs)
        lo = sum(1 for c in cs if c < LOW)/n*100
        mi = sum(1 for c in cs if LOW <= c < MID)/n*100
        hi = sum(1 for c in cs if c >= MID)/n*100
        rows_data.append((t, lo, mi, hi))

    W = 820
    H = 60+len(rows_data)*44+50
    left, right, top = 200, 40, 60
    plot_w = W-left-right
    s = [svg_header(W, H, "Konfidenz-Bänder je Dimension (Anteil der Labels)")]
    for i, (t, lo, mi, hi) in enumerate(rows_data):
        y = top+i*44
        s.append(f'<text x="{left-10}" y="{y+18}" font-size="13" text-anchor="end" fill="{DARK}">{esc(t)}</text>')
        x = left
        for val, col in [(lo, RED), (mi, ORANGE), (hi, GREEN)]:
            w = plot_w*val/100
            s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="26" fill="{col}"/>')
            if val >= 7:
                s.append(f'<text x="{x+w/2:.1f}" y="{y+17}" font-size="10" text-anchor="middle" fill="white">{val:.0f}%</text>')
            x += w
    # Legende
    ly = top+len(rows_data)*44+8
    for lx, (col, lab) in zip([left, left+180, left+360],
                              [(RED, "< 0.5 (Rateverdacht)"), (ORANGE, "0.5-0.7"), (GREEN, ">= 0.7 (sicher)")]):
        s.append(f'<rect x="{lx}" y="{ly-11}" width="12" height="12" fill="{col}"/>')
        s.append(f'<text x="{lx+18}" y="{ly}" font-size="11" fill="{DARK}">{esc(lab)}</text>')
    s.append("</svg>")
    path.write_text("".join(s), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--outdir",required=True,type=Path)
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)

    rows=load(args.input)
    if not rows: print("Keine Daten.",flush=True); sys.exit(1)
    n_comments=len(set(r.get("comment_id") for r in rows))
    print(f"Geladen: {len(rows)} Zeilen, {n_comments} Kommentare",flush=True)

    by_task=defaultdict(list)
    for r in rows: by_task[r.get("task")].append(r)
    tasks=[t for t in ALL_TASKS if t in by_task]

    labeled=[]; abstained=[]; errored=[]
    for t in tasks:
        vs=[val_of(r) for r in by_task[t]]
        labeled.append(sum(1 for v in vs if v not in ("ABSTAIN","ERROR")))
        abstained.append(sum(1 for v in vs if v=="ABSTAIN"))
        errored.append(sum(1 for v in vs if v=="ERROR"))

    stacked_bars(args.outdir/"01_abdeckung.svg", tasks, labeled, abstained, errored, n_comments)
    print("  geschrieben: 01_abdeckung.svg",flush=True)

    for i,(task,(title,ticks)) in enumerate(NUMERIC.items(),1):
        distribution_bars(args.outdir/f"02_{i}_{task}.svg", task, title, ticks, by_task)
        print(f"  geschrieben: 02_{i}_{task}.svg",flush=True)

    # Histogramme für kontinuierliche Dimensionen
    CONTINUOUS = {
        "agreement":             ("Agreement (-1=Widerspruch … 1=Zustimmung)", -1.0, 1.0, 10),
        "responsiveness":        ("Responsiveness (0=gar nicht … 1=voll)",      0.0, 1.0, 10),
        "epistemic_modality":    ("Epistemic Modality (0=sicher … 1=unsicher)", 0.0, 1.0, 10),
        "justification_density": ("Justification Density (pro 100 Wörter)",     0.0, 5.0, 10),
    }
    for j,(task,(title,lo,hi,nb)) in enumerate(CONTINUOUS.items(),1):
        if task in by_task:
            histogram(args.outdir/f"04_{j}_{task}.svg", task, title, lo, hi, nb, by_task)
            print(f"  geschrieben: 04_{j}_{task}.svg",flush=True)

    confidence_bars(args.outdir/"03_konfidenz.svg", tasks, by_task)
    print("  geschrieben: 03_konfidenz.svg",flush=True)

    # Neue Konfidenz-Diagramme
    confidence_hist_by_length(args.outdir/"05_konfidenz_kurz_lang.svg", rows)
    print("  geschrieben: 05_konfidenz_kurz_lang.svg",flush=True)

    confidence_hist_per_task(args.outdir/"06_konfidenz_baender.svg", tasks, by_task)
    print("  geschrieben: 06_konfidenz_baender.svg",flush=True)

    # Text-Tabelle
    with open(args.outdir/"zusammenfassung.txt","w",encoding="utf-8") as f:
        f.write(f"Labeling-Ergebnisse: {args.input.name}\n")
        f.write(f"Kommentare: {n_comments} | Ergebniszeilen: {len(rows)}\n\n")
        f.write(f"{'Dimension':24}{'gelabelt':>10}{'ABSTAIN':>10}{'Fehler':>8}{'Konfidenz':>11}\n")
        f.write("-"*63+"\n")
        for i,t in enumerate(tasks):
            cs=[conf_of(r) for r in by_task[t]]; cs=[c for c in cs if c is not None]
            mc=sum(cs)/len(cs) if cs else 0
            f.write(f"{t:24}{labeled[i]:>10}{abstained[i]:>10}{errored[i]:>8}{mc:>11.2f}\n")
    print("  geschrieben: zusammenfassung.txt",flush=True)
    print(f"\nFertig. Alles in: {args.outdir}",flush=True)
    print("SVG im Browser öffnen (Doppelklick) und in Folien einfügen.",flush=True)

if __name__=="__main__":
    main()
