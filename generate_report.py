# Generates a detailed evaluation report CSV from a batch results file.
# Usage: python generate_report.py batch_results_XXXXXXXX_XXXXXX.csv

import csv
import sys
from collections import defaultdict
from pathlib import Path

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "batch_results_20260428_080506.csv"
OUTPUT_FILE = Path(INPUT_FILE).stem + "_REPORT.csv"
CONFIDENCE_THRESHOLD = 0.70


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "N/A"


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("true_label")]  # skip blank/summary rows
    return rows


def main():
    rows = read_rows(INPUT_FILE)

    #enrich each row ───────────────────────────────────────────────────
    enriched = []
    for r in rows:
        conf = float(r["confidence"])
        top2 = float(r["top2_confidence"])
        accepted = conf >= CONFIDENCE_THRESHOLD
        top1_correct = accepted and r["predicted_label"] == r["true_label"]
        true_set = r["true_label"].split()[0] if r["true_label"] else ""
        enriched.append({
            "filename":             r["filename"],
            "true_label":           r["true_label"],
            "true_set":             true_set,
            "predicted_label":      r["predicted_label"],
            "confidence_%":         f"{conf*100:.1f}",
            "accepted_(≥70%)":      "Yes" if accepted else "No",
            "top1_correct":         "Yes" if top1_correct else "No",
            "top2_label":           r["top2_label"],
            "top2_confidence_%":    f"{top2*100:.1f}",
            "top3_label":           r["top3_label"],
            "top3_confidence_%":    f"{float(r['top3_confidence'])*100:.1f}",
            "confidence_margin_%":  f"{(conf - top2)*100:.1f}",
            "true_in_top3":         "Yes" if r["true_label"] in [r["predicted_label"], r["top2_label"], r["top3_label"]] else "No",
        })

    #per-card summary ──────────────────────────────────────────────────
    card_stats = defaultdict(lambda: {"set": "", "photos": 0, "accepted": 0,
                                      "top1": 0, "top3": 0})
    for r in enriched:
        c = r["true_label"]
        card_stats[c]["set"] = r["true_set"]
        card_stats[c]["photos"] += 1
        if r["accepted_(≥70%)"] == "Yes":
            card_stats[c]["accepted"] += 1
        if r["top1_correct"] == "Yes":
            card_stats[c]["top1"] += 1
        if r["true_in_top3"] == "Yes":
            card_stats[c]["top3"] += 1

    #per-set summary ───────────────────────────────────────────────────
    set_stats = defaultdict(lambda: {"cards": set(), "photos": 0, "accepted": 0,
                                     "top1": 0, "top3": 0})
    for r in enriched:
        s = r["true_set"]
        set_stats[s]["cards"].add(r["true_label"])
        set_stats[s]["photos"] += 1
        if r["accepted_(≥70%)"] == "Yes":
            set_stats[s]["accepted"] += 1
        if r["top1_correct"] == "Yes":
            set_stats[s]["top1"] += 1
        if r["true_in_top3"] == "Yes":
            set_stats[s]["top3"] += 1

    #overall ───────────────────────────────────────────────────────────
    total      = len(enriched)
    accepted   = sum(1 for r in enriched if r["accepted_(≥70%)"] == "Yes")
    top1_total = sum(1 for r in enriched if r["top1_correct"] == "Yes")
    top3_total = sum(1 for r in enriched if r["true_in_top3"] == "Yes")
    confs      = [float(r["confidence_%"]) for r in enriched]

    #write report ──────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)

        #Section 1: per-image results ──────────────────────────────────
        w.writerow(["=== SECTION 1: PER-IMAGE RESULTS ==="])
        w.writerow([
            "Filename",
            "True Label (correct card)",
            "Set",
            "Predicted Label",
            "Confidence %",
            "Accepted? (≥70% threshold)",
            "Top-1 Correct? (accepted & right)",
            "2nd Choice Label",
            "2nd Choice Confidence %",
            "3rd Choice Label",
            "3rd Choice Confidence %",
            "Confidence Margin % (1st minus 2nd)",
            "True Card in Top 3?",
        ])
        for r in sorted(enriched, key=lambda x: (x["true_set"], x["true_label"], x["filename"])):
            w.writerow([
                r["filename"],
                r["true_label"],
                r["true_set"],
                r["predicted_label"],
                r["confidence_%"],
                r["accepted_(≥70%)"],
                r["top1_correct"],
                r["top2_label"],
                r["top2_confidence_%"],
                r["top3_label"],
                r["top3_confidence_%"],
                r["confidence_margin_%"],
                r["true_in_top3"],
            ])

        #Section 2: per-card summary ───────────────────────────────────
        w.writerow([])
        w.writerow(["=== SECTION 2: PER-CARD SUMMARY ==="])
        w.writerow([
            "Card ID",
            "Set",
            "Photos Tested",
            "Photos Accepted",
            "Acceptance Rate",
            "Top-1 Correct (accepted & right)",
            "Top-1 Accuracy (of accepted)",
            "True Label in Top 3",
            "Top-3 Rate (of accepted)",
        ])
        for card, st in sorted(card_stats.items()):
            n = st["photos"]
            a = st["accepted"]
            w.writerow([
                card,
                st["set"],
                n,
                a,
                pct(a, n),
                st["top1"],
                pct(st["top1"], a),
                st["top3"],
                pct(st["top3"], a),
            ])

        #Section 3: per-set summary ────────────────────────────────────
        w.writerow([])
        w.writerow(["=== SECTION 3: PER-SET SUMMARY ==="])
        w.writerow([
            "Set",
            "Unique Cards Tested",
            "Total Photos",
            "Accepted (≥70%)",
            "Acceptance Rate",
            "Top-1 Correct",
            "Top-1 % of All Photos",
            "Top-1 % of Accepted",
            "True Label in Top 3",
            "Top-3 % of Accepted",
        ])
        for s, st in sorted(set_stats.items()):
            n = st["photos"]
            a = st["accepted"]
            w.writerow([
                s,
                len(st["cards"]),
                n,
                a,
                pct(a, n),
                st["top1"],
                pct(st["top1"], n),
                pct(st["top1"], a),
                st["top3"],
                pct(st["top3"], a),
            ])

        #Section 4: overall summary ────────────────────────────────────
        w.writerow([])
        w.writerow(["=== SECTION 4: OVERALL SUMMARY ==="])
        w.writerow(["Metric", "Value", "Notes"])
        w.writerow(["Model architecture",      "MobileNetV2 (transfer learning, ImageNet pretrained)"])
        w.writerow(["Input resolution",        "160 × 160 px"])
        w.writerow(["Confidence threshold",    "70%",        "Predictions below this are rejected"])
        w.writerow(["Target sets",             "JTG, SFA, SSP",  "3 of 5 sets had physical test cards available"])
        w.writerow(["Training images source",  "Pokémon TCG API (official card renders) + augmentation"])
        w.writerow([])
        w.writerow(["Unique cards tested",     len(card_stats)])
        w.writerow(["Photos per card",         "4"])
        w.writerow(["Total photos",            total])
        w.writerow([])
        w.writerow(["Photos accepted (≥70%)",  accepted,     pct(accepted, total)])
        w.writerow(["Photos rejected (<70%)",  total-accepted, pct(total-accepted, total)])
        w.writerow([])
        w.writerow(["Top-1 correct (of all photos)",    top1_total,  pct(top1_total, total)])
        w.writerow(["Top-1 correct (of accepted only)", top1_total,  pct(top1_total, accepted)])
        w.writerow(["True label in top 3 (of accepted)", top3_total, pct(top3_total, accepted)])
        w.writerow([])
        w.writerow(["Mean confidence (all photos)",  f"{sum(confs)/len(confs):.1f}%"])
        w.writerow(["Min confidence",  f"{min(confs):.1f}%"])
        w.writerow(["Max confidence",  f"{max(confs):.1f}%"])

    print(f"Report written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
