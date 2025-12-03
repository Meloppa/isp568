import io
from typing import Any, List, Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import time

def generate_student_report_pdf(
    inputs: Dict[str, float],
    evaluation: Dict[str, Any],
    suggestion: str
) -> io.BytesIO:
    """
    Generates a PDF report containing student evaluation, fuzzy score, and AI suggestion.
    Returns an in-memory BytesIO buffer containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    story: List[Any] = []
    
    # Title
    story.append(Paragraph("<b>Student Performance Evaluation Report</b>", styles['h1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 24))
    
    # 1. Input Data Table
    story.append(Paragraph("<b>1. Input Scores</b>", styles['h2']))
    data = [
        ["Metric", "Score (%)"],
        ["Attendance", f"{inputs['attendance']:.1f}"],
        ["Test Score", f"{inputs['test_score']:.1f}"],
        ["Assignment Score", f"{inputs['assignment_score']:.1f}"]
    ]
    
    table = Table(data, colWidths=[150, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#052C4D')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EFEFEF')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(table)
    story.append(Spacer(1, 24))
    
    # 2. Evaluation Result
    final_score = evaluation['fuzzy_score']
    level = evaluation['performance_level']
    
    story.append(Paragraph("<b>2. Evaluation Results (Fuzzy Logic)</b>", styles['h2']))
    story.append(Paragraph(f"<b>Fuzzy Performance Score:</b> {final_score:.2f} / 100", styles['Normal']))
    
    # Use color for performance level
    color = 'green' if level in ['Good', 'Excellent'] else ('orange' if level == 'Average' else 'red')
    story.append(Paragraph(
        f"<b>Performance Level:</b> <font color='{color}'>{level.upper()}</font>", 
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # 3. AI Suggestion
    story.append(Paragraph("<b>3. Academic Advisor's Suggestion</b>", styles['h2']))
    # Use Paragraph flowable and replace newlines with breaks
    story.append(Paragraph(suggestion.replace('\n', '<br/>'), styles['BodyText']))
    story.append(Spacer(1, 24))
    
    # Build the PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer