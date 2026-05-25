# Generates a formatted Excel workbook from a batch results CSV.
# Usage: python generate_excel.py batch_results_XXXXXXXX_XXXXXX.csv

import csv
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "batch_results_20260428_080506.csv"
OUTPUT_FILE = Path(INPUT_FILE).stem.replace("_REPORT", "") + "_EXCEL.xlsx"
CONFIDENCE_THRESHOLD = 0.70

#colours ───────────────────────────────────────────────────────────────────
HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")   # dark blue
SUBHDR_FILL   = PatternFill("solid", fgColor="2E75B6")   # mid blue
CORRECT_FILL  = PatternFill("solid", fgColor="C6EFCE")   # green
WRONG_FILL    = PatternFill("solid", fgColor="FFCCCC")   # red
REJECT_FILL   = PatternFill("solid", fgColor="FFF2CC")   # yellow
ALT_FILL      = PatternFill("solid", fgColor="F2F2F2")   # light grey
WHITE_FILL    = PatternFill("solid", fgColor="FFFFFF")

HEADER_FONT  = Font(bold=True, color="FFFFFF", size=11)
SUBHDR_FONT  = Font(bold=True, color="FFFFFF", size=10)
BOLD_FONT    = Font(bold=True, size=10)
NORMAL_FONT  = Font(size=10)

THIN = Side(style="thin", color="CCCCCC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT   = Alignment(horizontal="left",   vertical="center", wrap_text=False)


def read_raw_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("true_label") and r.get("confidence")]


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "N/A"


def style_header(ws, row, col_count, fill=None, font=None, height=28):
    ws.row_dimensions[row].height = height
    for col in range(1, col_count + 1):
        cell = ws.cell(row=row, col=col)
        if fill:
            cell.fill = fill
        if font:
            cell.font = font
        cell.border = BORDER
        cell.alignment = CENTER


def write_header_row(ws, row_num, headers, fill, font):
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=row_num, column=col, value=h)
        cell.fill = fill
        cell.font = font
        cell.border = BORDER
        cell.alignment = CENTER
    ws.row_dimensions[row_num].height = 30


def write_data_row(ws, row_num, values, fill=None, font=None, alignments=None):
    for col, v in enumerate(values, 1):
        cell = ws.cell(row=row_num, column=col, value=v)
        cell.fill = fill or WHITE_FILL
        cell.font = font or NORMAL_FONT
        cell.border = BORDER
        align = alignments[col - 1] if alignments and col - 1 < len(alignments) else LEFT
        cell.alignment = align
    ws.row_dimensions[row_num].height = 18


def auto_width(ws, min_w=8, max_w=40):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max((len(str(c.value or "")) for c in col), default=0)
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, min_w), max_w)


#build enriched row dicts ──────────────────────────────────────────────────
def enrich(rows):
    out = []
    for r in rows:
        conf    = float(r["confidence"])
        top2    = float(r["top2_confidence"])
        top3c   = float(r["top3_confidence"])
        accepted = conf >= CONFIDENCE_THRESHOLD
        correct  = accepted and r["predicted_label"] == r["true_label"]
        in_top3  = r["true_label"] in [r["predicted_label"], r["top2_label"], r["top3_label"]]
        true_set = r["true_label"].split()[0] if r["true_label"] else ""
        out.append({
            "filename":       r["filename"],
            "true_label":     r["true_label"],
            "true_set":       true_set,
            "predicted":      r["predicted_label"],
            "conf":           conf,
            "accepted":       accepted,
            "correct":        correct,
            "in_top3":        in_top3,
            "top2_label":     r["top2_label"],
            "top2_conf":      top2,
            "top3_label":     r["top3_label"],
            "top3_conf":      top3c,
            "margin":         conf - top2,
        })
    return out


#Sheet 1: per-image results ────────────────────────────────────────────────
def sheet_per_image(wb, rows):
    ws = wb.create_sheet("Per-Image Results")

    headers = [
        "Filename", "True Card", "Set",
        "Predicted Card", "Confidence %", "Accepted?",
        "Correct?", "2nd Choice", "2nd Conf %",
        "3rd Choice", "3rd Conf %", "Confidence Margin %",
        "True Card in Top 3?",
    ]
    write_header_row(ws, 1, headers, HEADER_FILL, HEADER_FONT)
    ws.freeze_panes = "A2"

    aligns = [LEFT, LEFT, CENTER, LEFT, CENTER, CENTER, CENTER,
              LEFT, CENTER, LEFT, CENTER, CENTER, CENTER]

    for i, r in enumerate(sorted(rows, key=lambda x: (x["true_set"], x["true_label"], x["filename"])), 2):
        if r["correct"]:
            fill = CORRECT_FILL
        elif r["accepted"]:
            fill = WRONG_FILL
        else:
            fill = REJECT_FILL

        vals = [
            r["filename"],
            r["true_label"],
            r["true_set"],
            r["predicted"],
            round(r["conf"] * 100, 1),
            "Yes" if r["accepted"] else "No",
            "Yes" if r["correct"] else "No",
            r["top2_label"],
            round(r["top2_conf"] * 100, 1),
            r["top3_label"],
            round(r["top3_conf"] * 100, 1),
            round(r["margin"] * 100, 1),
            "Yes" if r["in_top3"] else "No",
        ]
        write_data_row(ws, i, vals, fill=fill, alignments=aligns)

    # legend
    legend_row = len(rows) + 3
    ws.cell(row=legend_row, column=1, value="Colour key:").font = BOLD_FONT
    for offset, (label, fill) in enumerate([
        ("Accepted & correct", CORRECT_FILL),
        ("Accepted & wrong", WRONG_FILL),
        ("Rejected (<70% confidence)", REJECT_FILL),
    ], 1):
        c = ws.cell(row=legend_row + offset, column=1, value=label)
        c.fill = fill
        c.font = NORMAL_FONT
        c.border = BORDER

    auto_width(ws)


#Sheet 2: per-card summary ─────────────────────────────────────────────────
def sheet_per_card(wb, rows):
    ws = wb.create_sheet("Per-Card Summary")

    card_stats = defaultdict(lambda: {"set": "", "photos": 0, "accepted": 0, "top1": 0, "top3": 0})
    for r in rows:
        c = r["true_label"]
        card_stats[c]["set"] = r["true_set"]
        card_stats[c]["photos"] += 1
        if r["accepted"]:  card_stats[c]["accepted"] += 1
        if r["correct"]:   card_stats[c]["top1"] += 1
        if r["in_top3"]:   card_stats[c]["top3"] += 1

    headers = [
        "Card ID", "Set", "Photos Tested", "Photos Accepted",
        "Acceptance Rate", "Correct Predictions",
        "Accuracy (of Accepted)", "True Card in Top 3", "Top-3 Rate (of Accepted)",
    ]
    write_header_row(ws, 1, headers, HEADER_FILL, HEADER_FONT)
    ws.freeze_panes = "A2"

    aligns = [LEFT, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER, CENTER]

    for i, (card, st) in enumerate(sorted(card_stats.items()), 2):
        n, a = st["photos"], st["accepted"]
        t1, t3 = st["top1"], st["top3"]
        acc = t1 / a if a else 0
        fill = CORRECT_FILL if acc >= 0.7 else (WRONG_FILL if a > 0 else REJECT_FILL)
        write_data_row(ws, i, [
            card, st["set"], n, a,
            pct(a, n), t1,
            pct(t1, a), t3,
            pct(t3, a),
        ], fill=fill, alignments=aligns)

    auto_width(ws)


#Sheet 3: per-set + overall ────────────────────────────────────────────────
def sheet_summary(wb, rows):
    ws = wb.create_sheet("Summary")

    set_stats = defaultdict(lambda: {"cards": set(), "photos": 0, "accepted": 0, "top1": 0, "top3": 0})
    for r in rows:
        s = r["true_set"]
        set_stats[s]["cards"].add(r["true_label"])
        set_stats[s]["photos"] += 1
        if r["accepted"]: set_stats[s]["accepted"] += 1
        if r["correct"]:  set_stats[s]["top1"] += 1
        if r["in_top3"]:  set_stats[s]["top3"] += 1

    row = 1
    ws.cell(row=row, column=1, value="Per-Set Breakdown").font = Font(bold=True, size=13)
    row += 1

    set_headers = [
        "Set", "Cards Tested", "Photos", "Accepted (>=70%)",
        "Acceptance Rate", "Top-1 Correct", "Top-1 % (all photos)",
        "Top-1 % (accepted only)", "True in Top 3", "Top-3 % (accepted only)",
    ]
    write_header_row(ws, row, set_headers, SUBHDR_FILL, SUBHDR_FONT)
    ws.freeze_panes = f"A{row + 1}"
    row += 1

    aligns = [CENTER] * len(set_headers)
    for s, st in sorted(set_stats.items()):
        n, a = st["photos"], st["accepted"]
        t1 = st["top1"]
        fill = ALT_FILL if row % 2 == 0 else WHITE_FILL
        write_data_row(ws, row, [
            s, len(st["cards"]), n, a,
            pct(a, n), t1,
            pct(t1, n), pct(t1, a),
            st["top3"], pct(st["top3"], a),
        ], fill=fill, alignments=aligns)
        row += 1

    row += 2
    total    = len(rows)
    accepted = sum(1 for r in rows if r["accepted"])
    top1     = sum(1 for r in rows if r["correct"])
    top3     = sum(1 for r in rows if r["in_top3"])
    confs    = [r["conf"] for r in rows]

    ws.cell(row=row, column=1, value="Overall Summary").font = Font(bold=True, size=13)
    row += 1

    overall_data = [
        ("Model",                    "MobileNetV2 (transfer learning, ImageNet pretrained)"),
        ("Input resolution",         "160 x 160 pixels"),
        ("Confidence threshold",     "70%  (predictions below this are rejected)"),
        ("Target sets in scope",     "JTG, PRE, SCR, SFA, SSP"),
        ("Sets with physical cards tested", "JTG, SFA, SSP"),
        ("Training images source",   "Pokemon TCG API + 15 augmented variants per card"),
        ("",                         ""),
        ("Unique cards tested",      len(set(r["true_label"] for r in rows))),
        ("Photos per card",          4),
        ("Total photos",             total),
        ("",                         ""),
        ("Photos accepted (>=70%)",  f"{accepted} / {total}  ({pct(accepted, total)})"),
        ("Photos rejected (<70%)",   f"{total - accepted} / {total}  ({pct(total - accepted, total)})"),
        ("",                         ""),
        ("Top-1 correct (of all photos)",    f"{top1} / {total}  ({pct(top1, total)})"),
        ("Top-1 correct (of accepted only)", f"{top1} / {accepted}  ({pct(top1, accepted)})"),
        ("True label in top 3 (accepted)",   f"{top3} / {accepted}  ({pct(top3, accepted)})"),
        ("",                         ""),
        ("Mean confidence (all photos)", f"{sum(confs)/len(confs)*100:.1f}%"),
        ("Min confidence",           f"{min(confs)*100:.1f}%"),
        ("Max confidence",           f"{max(confs)*100:.1f}%"),
    ]

    for label, value in overall_data:
        lc = ws.cell(row=row, column=1, value=label)
        vc = ws.cell(row=row, column=2, value=value)
        if label and label != "":
            lc.font = BOLD_FONT
            lc.fill = ALT_FILL if row % 2 == 0 else WHITE_FILL
            vc.fill = ALT_FILL if row % 2 == 0 else WHITE_FILL
        lc.border = BORDER
        vc.border = BORDER
        lc.alignment = LEFT
        vc.alignment = LEFT
        row += 1

    auto_width(ws)
    ws.column_dimensions["B"].width = 55


#public API ────────────────────────────────────────────────────────────────
def build_excel(csv_path, excel_path):
    """Read a batch-results CSV and write a formatted Excel workbook."""
    raw  = read_raw_rows(csv_path)
    rows = enrich(raw)

    wb = Workbook()
    wb.remove(wb.active)

    sheet_per_image(wb, rows)
    sheet_per_card(wb, rows)
    sheet_summary(wb, rows)

    wb.save(excel_path)


#CLI entry point ────────────────────────────────────────────────────────────
def main():
    raw  = read_raw_rows(INPUT_FILE)
    rows = enrich(raw)

    wb = Workbook()
    wb.remove(wb.active)

    sheet_per_image(wb, rows)
    sheet_per_card(wb, rows)
    sheet_summary(wb, rows)

    wb.save(OUTPUT_FILE)
    print(f"Excel report written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
