"""Read local uploads into labelled passages for retrieval."""
import io
import re
import zipfile
from pathlib import Path

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
                rows = enumerate(sheet.values, 1)
                header = ""
                group = []
                start = 1
                for i, row in rows:
                    values = " | ".join("" if x is None else str(x) for x in row).strip(" |")
                    if not values:
                        continue
                    if not header:
                        header = values[:300]
                    line = f"Row {i}: {values}"
                    if group and (sum(map(len, group)) + len(line) > 800 or len(group) >= 8):
                        yield f"{sheet.title}, rows {start}–{i-1}", f"Columns: {header}\n" + "\n".join(group)
                        group = []
                    if not group:
                        start = i
                    group.append(line)
                if group:
                    yield f"{sheet.title}, from row {start}", f"Columns: {header}\n" + "\n".join(group)
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

def extract(name, data, *, pages=None):
    chunks = []
    total = 0
    for label,text in (passages(name,data) if pages is None else pages):
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



def student_roster(path):
    """Read the roster from a workbook with one clearly identified student sheet."""
    from openpyxl import load_workbook

    counts = []
    with path.open("rb") as source:
        book = load_workbook(source, read_only=True, data_only=True)
        try:
            for sheet in book:
                rows = iter(sheet.values)
                header = next((row for row in rows if any(value is not None for value in row)), ())
                columns = [str(value).strip().casefold() for value in header]
                if "student id" in columns:
                    index = columns.index("student id")
                    name_index = columns.index("student name") if "student name" in columns else None
                    ids, names = [], []
                    for row in rows:
                        if index >= len(row) or row[index] is None or not str(row[index]).strip():
                            continue
                        ids.append(str(row[index]).strip())
                        if name_index is not None and name_index < len(row) and row[name_index] is not None:
                            names.append(str(row[name_index]).strip())
                    counts.append((sheet.title, len(ids), len(set(ids)), names))
        finally:
            book.close()
    return counts[0] if len(counts) == 1 else None
