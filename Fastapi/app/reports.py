import io
from typing import Any, List, Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import time

def generate_student_report_pdf(
    inputs: Dict[str, float],
    evaluation: Dict[str, Any],
    suggestion: str
) -> io.BytesIO:
    """
    Generates a university-style PDF report containing student evaluation,
    fuzzy score, applied rules, and AI suggestion.
    Returns an in-memory BytesIO buffer containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='TitleCenter', fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='Heading', fontSize=14, leading=18, spaceAfter=12, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='NormalLeft', fontSize=11, leading=14, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='SmallCenter', fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.grey))
    
    story: List[Any] = []
    
    # University Header
    story.append(Paragraph("UNIVERSITY NAME / FACULTY", styles['TitleCenter']))
    story.append(Paragraph("STUDENT PERFORMANCE EVALUATION REPORT", styles['TitleCenter']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Report Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['SmallCenter']))
    story.append(Spacer(1, 24))
    
    # 1. Student Input Scores
    story.append(Paragraph("1. Student Input Scores", styles['Heading']))
    data = [
        ["Metric", "Score (%)"],
        ["Attendance", f"{inputs['attendance']:.1f}"],
        ["Test Score", f"{inputs['test_score']:.1f}"],
        ["Assignment Score", f"{inputs['assignment_score']:.1f}"],
        ["Ethics", f"{inputs.get('ethics', 0.0):.1f}"],
        ["Cognitive Skills", f"{inputs.get('cognitive', 0.0):.1f}"]
    ]
    
    table = Table(data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
    ]))
    story.append(table)
    story.append(Spacer(1, 24))
    
    # 2. Fuzzy Evaluation Results
    story.append(Paragraph("2. Evaluation Results (Fuzzy Logic)", styles['Heading']))
    final_score = evaluation['fuzzy_score']
    level = evaluation['performance_level']
    color = 'green' if level in ['Good', 'Excellent'] else ('orange' if level == 'Average' else 'red')
    
    story.append(Paragraph(f"<b>Fuzzy Score:</b> {final_score:.2f} / 100", styles['NormalLeft']))
    story.append(Paragraph(f"<b>Performance Level:</b> <font color='{color}'>{level.upper()}</font>", styles['NormalLeft']))
    story.append(Spacer(1, 12))
    
    # 3. Applied Rules
    rules = evaluation.get('applied_rules', [])
    if rules:
        story.append(Paragraph("3. Applied Fuzzy Rules", styles['Heading']))
        rules_data = [["Rule ID", "IF…THEN Logic", "Strength"]]
        for r in rules:
            rules_data.append([r['rule_id'], r['logic'], f"{r['strength']:.2f}"])
        
        rules_table = Table(rules_data, colWidths=[60, 320, 60])
        rules_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9F9F9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
        ]))
        story.append(rules_table)
        story.append(Spacer(1, 24))
    
    # 4. AI Suggestion
    story.append(Paragraph("4. Academic Advisor's Suggestion", styles['Heading']))
    story.append(Paragraph(suggestion.replace('\n', '<br/>'), styles['NormalLeft']))
    story.append(Spacer(1, 24))
    
    # Footer / Disclaimer
    story.append(Paragraph("Note: This evaluation is based on fuzzy logic analysis and AI-generated academic suggestions.", styles['SmallCenter']))
    
    # Build PDF
    doc.build(story)
    
    buffer.seek(0)
    return buffer
