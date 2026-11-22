"""In-memory exports for a conversation, one reply, or one rendered table."""
import io
import json
from html import escape

def export_file(request):
    rows = request.rows
    turns = [m.model_dump() for m in request.messages]
    title = "Table" if rows else "Conversation"
    if request.format == "json":
        return json.dumps(rows or turns, ensure_ascii=False, indent=2).encode(), "application/json"
    stream = io.BytesIO()
    if request.format == "xlsx":
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
        book = Workbook()
        sheet = book.active
        sheet.title = title
        values = rows or [["Role", "Message", "Sources"]] + [
            [t["role"], t["text"], "\n".join(t["sources"])] for t in turns]
        for row in values:
            sheet.append(row)
            for cell in sheet[sheet.max_row]:
                if cell.value is not None:
                    cell.data_type = "s"
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if cell.row == 1:
                    cell.font = Font(bold=True)
        sheet.freeze_panes = "A2"
        for i in range(sheet.max_column):
            sheet.column_dimensions[get_column_letter(i+1)].width = 50
        book.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if request.format == "docx":
        from docx import Document
        doc = Document()
        doc.add_heading(title, 0)
        if rows:
            table = doc.add_table(rows=0, cols=max(map(len, rows)))
            table.style = "Table Grid"
            for row in rows:
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    cell.text = value
        for t in turns:
            doc.add_heading(t["role"].title(), 2)
            doc.add_paragraph(t["text"])
            for url in t["sources"]:
                doc.add_paragraph(url)
        doc.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    import pymupdf
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    html = "<h1>" + title + "</h1>"
    if rows:
        html += "<table>" + "".join("<tr>" + "".join("<td>"+escape(c)+"</td>" for c in row)+"</tr>" for row in rows)+"</table>"
    for t in turns:
        html += "<h2>"+t["role"].title()+"</h2>"+md.render(t["text"])
        html += "".join("<p>"+escape(url)+"</p>" for url in t["sources"])
    story = pymupdf.Story(html=html, user_css="td,th{border:1px solid #bbb;padding:4px} body{font-family:sans-serif}")
    page = pymupdf.paper_rect("a4")
    writer = pymupdf.DocumentWriter(stream)
    story.write(writer, lambda *_: (page, page + (40,40,-40,-40), None))
    writer.close()
    return stream.getvalue(), "application/pdf"

