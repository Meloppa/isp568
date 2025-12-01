from reportlab.pdfgen import canvas

def generate_pdf(name, score, category, feedback):
    filename = f"{name}_report.pdf"
    c = canvas.Canvas(filename)

    c.drawString(50, 800, f"Student Report: {name}")
    c.drawString(50, 760, f"Total Score: {score}")
    c.drawString(50, 740, f"Category: {category}")
    c.drawString(50, 700, "Feedback:")

    y = 680
    for item in feedback:
        c.drawString(60, y, f"- {item}")
        y -= 20

    c.save()
    return filename
