import argparse
import csv
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate

from eta_model import fit_and_evaluate


def load_shipments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_pdf(source: Path, dataset: Path, output: Path) -> None:
    _, metrics = fit_and_evaluate(load_shipments(dataset))
    replacements = {
        "test_shipments": f"{metrics['test_shipments']:.0f}",
        "mae_hours": f"{metrics['mae_hours']:.2f}",
        "baseline_mae_hours": f"{metrics['baseline_mae_hours']:.2f}",
        "late_accuracy": f"{metrics['late_accuracy']:.3f}",
        "late_baseline_accuracy": f"{metrics['late_baseline_accuracy']:.3f}",
        "late_precision": f"{metrics['late_precision']:.3f}",
        "late_recall": f"{metrics['late_recall']:.3f}",
        "late_f1": f"{metrics['late_f1']:.3f}",
    }
    report_text = source.read_text(encoding="utf-8")
    for key, value in replacements.items():
        report_text = report_text.replace("{{" + key + "}}", value)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=colors.HexColor("#173B43"),
        alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=colors.HexColor("#176B67"),
        spaceBefore=9, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13, textColor=colors.HexColor("#202B2C"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["BodyText"], fontName="Helvetica-Oblique",
        fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#526367"),
        spaceAfter=12,
    ))

    story = []
    paragraph_lines = []
    bullet_items = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = escape(" ".join(paragraph_lines))
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            story.append(Paragraph(text, styles["ReportBody"]))
            paragraph_lines.clear()

    def flush_bullets() -> None:
        if bullet_items:
            items = [ListItem(Paragraph(escape(item), styles["ReportBody"])) for item in bullet_items]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=15, bulletFontSize=7))
            bullet_items.clear()

    for line in report_text.splitlines():
        if line == "---PAGE BREAK---":
            flush_paragraph()
            flush_bullets()
            story.append(PageBreak())
        elif line.startswith("# "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(escape(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(escape(line[3:]), styles["SectionHeading"]))
        elif line.startswith("**") and line.endswith("**"):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(escape(line[2:-2]), styles["ReportSubtitle"]))
        elif line.startswith("- "):
            flush_paragraph()
            bullet_items.append(line[2:])
        elif not line.strip():
            flush_paragraph()
            flush_bullets()
        else:
            flush_bullets()
            paragraph_lines.append(line.strip())
    flush_paragraph()
    flush_bullets()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm, title="Predicting Shipment ETA and Delay Propagation",
        author="AIQON Take-Home Assignment",
    )

    def add_page_number(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6E0DF"))
        canvas.line(19 * mm, 13 * mm, A4[0] - 19 * mm, 13 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526367"))
        canvas.drawString(19 * mm, 8.5 * mm, "Shipment ETA and Delay Propagation")
        canvas.drawRightString(A4[0] - 19 * mm, 8.5 * mm, f"{doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Created {output} ({metrics['test_shipments']:.0f} test shipments, MAE {metrics['mae_hours']:.2f} h)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the technical report PDF with current evaluation metrics.")
    parser.add_argument("--source", type=Path, default=Path("report.md"))
    parser.add_argument("--data", type=Path, default=Path("data/shipments.csv"))
    parser.add_argument("--output", type=Path, default=Path("report.pdf"))
    args = parser.parse_args()
    build_pdf(args.source, args.data, args.output)


if __name__ == "__main__":
    main()