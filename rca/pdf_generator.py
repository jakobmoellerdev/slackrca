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
        self.multi_cell(0, 10, body.encode('latin-1', 'replace').decode('latin-1'))  # Ensure UTF-8 to Latin-1 conversion
        self.ln()

def generate_pdf(templated, raw, thread, filename):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title('Generation Disclaimer')
    pdf.chapter_body('This document was automatically generated by an AI system and may contain inaccuracies.'
                     'Please review the content carefully before taking any action.'
                     'The AI system is not responsible for any decisions made based on the information provided.'
                     'For any questions or concerns, please contact the system administrator.'
                     'This document is for internal use only and should not be shared externally.'
                     'Any potentially confidential information should be handled with care.')

    pdf.add_page()
    pdf.write_html(templated)

    pdf.add_page()
    pdf.chapter_title('Raw Output for Reference')
    pdf.write_html(raw)

    pdf.add_page()
    pdf.chapter_title('Conversation Information for Reference')
    pdf.chapter_body(f'Channel ID: {thread.channel_id}')
    pdf.chapter_body(f'Thread Timestamp: {thread.thread_ts}')
    pdf.chapter_body(f'Messages: {thread.messages}')

    pdf.output(filename)
