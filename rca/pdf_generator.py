# rca/pdf_generator.py
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Root Cause Analysis Summary', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def generate_pdf(summary, filename):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("Summary of Root Cause Analysis")
    pdf.set_text_color(255, 0, 0)  # Highlighting text in red
    pdf.chapter_body(summary)
    pdf.output(filename)
