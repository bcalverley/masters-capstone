# Generates a formatted PDF receipt from a BuyingSession.

from datetime import datetime

from fpdf import FPDF


#colour palette ─────────────────────────────────────────────────────────────
DARK_BLUE  = (31,  78, 121)
MID_BLUE   = (46, 117, 182)
LIGHT_BLUE = (222, 235, 247)
ROW_ALT    = (245, 245, 245)
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GREY       = (120, 120, 120)
GREEN      = (46, 125,  50)


class _ReceiptPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")


def export_receipt_pdf(session, filepath: str) -> None:
    """Write a formatted PDF receipt for *session* to *filepath*."""
    pdf = _ReceiptPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(left=12, top=12, right=12)
    pdf.add_page()

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin   # ≈ 186 mm

    #header banner ─────────────────────────────────────────────────────
    pdf.set_fill_color(*DARK_BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(usable_w, 13, "Pokemon Card Buying Receipt", align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(*MID_BLUE)
    pdf.cell(usable_w, 8,
             datetime.now().strftime("%A, %B %d, %Y   |   %I:%M %p"),
             align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    #column layout ─────────────────────────────────────────────────────
    #  #   Name   Set   No.   Variant   Grade   Price
    col_w = [9, 62, 14, 13, 30, 30, 28]
    headers = ["#", "Card Name", "Set", "No.", "Variant", "Grade", "Market Price"]
    aligns  = ["C", "L",  "C", "C", "C", "C", "R"]

    # table header row
    pdf.set_fill_color(*MID_BLUE)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    for h, w, a in zip(headers, col_w, aligns):
        pdf.cell(w, 7, h, border=1, fill=True, align=a)
    pdf.ln()

    # data rows
    pdf.set_font("Helvetica", "", 9)
    for i, item in enumerate(session.items):
        fill_color = ROW_ALT if i % 2 == 0 else WHITE
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*BLACK)

        price_str = f"${item.market_price:.2f}" if item.market_price else "N/A"
        name = item.card_name if item.card_name else "—"

        row    = [str(i + 1), name, item.set_code, item.card_number,
                  item.variant, item.grade, price_str]
        for val, w, a in zip(row, col_w, aligns):
            pdf.cell(w, 7, str(val), border=1, fill=True, align=a)
        pdf.ln()

    pdf.ln(5)

    #totals block ──────────────────────────────────────────────────────
    label_w = sum(col_w) - 36
    value_w = 36

    def totals_row(label, value, bold=False, text_color=BLACK, fill_color=LIGHT_BLUE):
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*text_color)
        pdf.set_font("Helvetica", "B" if bold else "", 10)
        pdf.cell(label_w, 8, label, border=1, fill=True, align="R")
        pdf.cell(value_w, 8, value, border=1, fill=True, align="R")
        pdf.ln()

    market_total = session.market_total()
    totals_row("Market Total", f"${market_total:.2f}",
               bold=True, text_color=DARK_BLUE, fill_color=(210, 228, 244))

    for pct in session.offer_tiers:
        totals_row(f"Offer at {pct}%", f"${session.offer_amount(pct):.2f}",
                   text_color=GREEN)

    #card count note ───────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    count = len(session.items)
    pdf.cell(0, 6, f"{count} card{'s' if count != 1 else ''} in this session.", align="L")

    pdf.output(filepath)
