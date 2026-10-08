"""Generate a small 3-page sample PDF with known facts, for tests."""

import os

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

PAGES = [
    (
        "SolarX-2000 Specifications",
        "The SolarX-2000 solar panel has an efficiency rating of 23.8%. "
        "It comes with a 25-year performance warranty and operates in "
        "temperatures from -40 to 85 degrees Celsius. "
        "Installation requires a south-facing roof with at least 20 square "
        "meters of unshaded area. The panel weighs 21.5 kilograms and uses "
        "monocrystalline silicon cells. Annual degradation is rated at "
        "0.4 percent per year.",
    ),
    (
        "AquaPure Filter Manual",
        "The AquaPure water filter removes 99.9% of contaminants including "
        "lead, chlorine, and microplastics. The filter cartridge needs "
        "replacement every 6 months under normal household use. "
        "A red indicator light signals when replacement is due. "
        "The unit fits standard 10-inch housings and handles a flow rate "
        "of 8 liters per minute.",
    ),
    (
        "Company History",
        "The company was founded in 2019 in Pune by Priya Sharma. "
        "It started as a two-person workshop building solar prototypes. "
        "By 2023 it employed over 200 people across three factories. "
        "Its mission is affordable clean technology for Indian households.",
    ),
]


def make_sample_pdf(out_dir: str) -> str:
    """Create sample.pdf in out_dir and return its path."""
    path = os.path.join(out_dir, "sample.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER
    for title, body in PAGES:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 72, title)
        c.setFont("Helvetica", 11)
        y = height - 110
        words, line = body.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 85:
                c.drawString(72, y, line)
                y -= 16
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            c.drawString(72, y, line)
        c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    import tempfile

    print(make_sample_pdf(tempfile.mkdtemp()))
