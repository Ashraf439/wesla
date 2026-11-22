"""Read local uploads into labelled passages for retrieval."""
import io
import re
import zipfile
from pathlib import Path
from collections import Counter

def passages(name, data):
    ext = Path(name).suffix.lower()
    file = io.BytesIO(data)
    if ext in {".docx", ".xlsx", ".pptx"}:
        with zipfile.ZipFile(file) as archive:
            if sum(x.file_size for x in archive.infolist()) > 80_000_000:
                raise ValueError("This Office document is too large to expand.")
        file.seek(0)
    if ext == ".pdf":
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("Unlock this PDF before uploading.")
            yield from ((f"page {p.number+1}", p.get_text()) for p in doc)
    elif ext == ".docx":
        from docx import Document
        doc = Document(file)
        for block in doc.iter_inner_content():
            if hasattr(block, "rows"):
                yield "table", "\n".join(" | ".join(c.text for c in row.cells) for row in block.rows)
            else:
                yield "paragraph", block.text
    elif ext == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(file, read_only=True, data_only=True)
        try:
            for sheet in book:
                for i,row in enumerate(sheet.values,1):
                    yield f"{sheet.title}, row {i}", " | ".join(str(x or "") if x != 0 else "0" for x in row)
        finally:
            book.close()
    elif ext == ".pptx":
        from lxml import etree
        with zipfile.ZipFile(file) as z:
            slides = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml",n)]
            for n in sorted(slides,key=lambda n:int(re.search(r"\d+",n).group())):
                root = etree.fromstring(z.read(n),parser=etree.XMLParser(resolve_entities=False, no_network=True))
                yield Path(n).stem, " ".join(root.xpath("//*[local-name()='t']/text()"))
    elif ext in {".txt",".md",".csv",".tsv",".json",".log",".py",".js",".html",".xml",".yaml",".yml"}:
        encoding = "utf-16" if data.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig"
        yield "text", data.decode(encoding)
    else:
        raise ValueError("Unsupported format. Use PDF, DOCX, XLSX, PPTX or a text file.")

def extract(name, data):
    chunks = []
    total = 0
    for label,text in passages(name,data):
        total += len(text)
        if total > 600000:
            raise ValueError("Please split files containing over 600,000 characters.")
        for start in range(0,len(text),1200):
            part = text[start:start+1200].strip()
            if part:
                chunks.append({"source":f"{name} · {label}", "text":part})
    if not chunks:
        raise ValueError("No readable text found. Scanned documents need OCR first.")
    return chunks

def select_passages(files, question):
    terms = set(re.findall(r"\w{3,}",question.lower()))
    candidates = []
    for f in files:
        ranked = sorted(f.chunks, key=lambda c: len(terms.intersection(re.findall(r"\w+",c.text.lower()))), reverse=True)
        candidates.append(ranked[:8])
    chosen = []
    for i in range(8):
        for group in candidates:
            if len(group)>i:
                chosen.append(group[i])
    return chosen

