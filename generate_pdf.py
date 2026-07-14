#!/usr/bin/env python3
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER

# Read markdown
with open('doc/project_reference.md', 'r') as f:
    content = f.read()

# Strip YAML
lines = content.split('\n')
end_idx = 0
yaml_count = 0
for i, line in enumerate(lines):
    if line.strip() == '---':
        yaml_count += 1
        if yaml_count == 2:
            end_idx = i
            break

markdown_content = '\n'.join(lines[end_idx+1:])

# Create PDF
doc = SimpleDocTemplate("doc/project_reference.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Add title
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor='#1a1a1a',
    spaceAfter=12,
    alignment=TA_CENTER
)
story.append(Paragraph("Subword Tokenizer & Prototype Training", title_style))
story.append(Spacer(1, 12))
story.append(Paragraph("<b>Production-validated 35.2M-param TinyLM (0.0107 eval loss)</b>", styles['Normal']))
story.append(Spacer(1, 6))
story.append(Paragraph("July 8, 2026", styles['Normal']))
story.append(PageBreak())

# Process markdown content
for line in markdown_content.split('\n'):
    if not line.strip():
        story.append(Spacer(1, 6))
    elif line.startswith('##'):
        level = len(line) - len(line.lstrip('#'))
        text = line.lstrip('#').strip()
        if level == 2:
            story.append(Paragraph(f"<b>{text}</b>", styles['Heading2']))
        elif level == 3:
            story.append(Paragraph(f"<i>{text}</i>", styles['Heading3']))
        else:
            story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(line, styles['Normal']))

doc.build(story)
print("PDF generated successfully: doc/project_reference.pdf")
