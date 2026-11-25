"""Read-only previews of locally stored uploads."""
import csv
import io
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from .attachments import passages

MEDIA = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".avif": "image/avif",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
}
TEXT = {
    ".txt", ".md", ".json", ".jsonl", ".log", ".py", ".js", ".ts", ".jsx",
    ".html", ".css", ".xml", ".yaml", ".yml", ".toml", ".ini", ".sql", ".rs",
    ".go", ".java", ".c", ".cpp", ".sh", ".ps1",
}
ROWS = 60
COLUMNS = 30

def table_preview(rows, part, **details):
    sample = list(islice(rows, ROWS + 1))
    return dict(kind="table", part=part, first_row=part * ROWS + 1,
                rows=[["" if cell is None else str(cell)[:1000] for cell in row[:COLUMNS]]
                      for row in sample[:ROWS]],
                more=len(sample) > ROWS, clipped_columns=any(len(row) > COLUMNS for row in sample),
                **details)

def describe(name, path, part=0, sheet=0):
    ext = Path(name).suffix.lower()
    if ext in MEDIA:
        return {"kind": MEDIA[ext].split("/")[0]}
    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(path, filetype="pdf") as doc:
            if doc.needs_pass:
                return dict(kind="unavailable", reason="Unlock this PDF to preview its pages.")
            return dict(kind="pdf", pages=len(doc))
    if ext == ".xlsx":
        from openpyxl import load_workbook

        with ZipFile(path) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 80_000_000:
                raise ValueError("Workbook is too large to expand.")
        with path.open("rb") as stream:
            book = load_workbook(stream, read_only=True, data_only=True)
            try:
                if sheet >= len(book.sheetnames):
                    raise ValueError("Worksheet does not exist.")
                tab = book.worksheets[sheet]
                rows = tab.iter_rows(min_row=part * ROWS + 1, values_only=True)
                return table_preview(rows, part, sheets=book.sheetnames, sheet=sheet)
            finally:
                book.close()
    if ext in {".csv", ".tsv"}:
        with path.open("rb") as raw:
            encoding = "utf-16" if raw.read(2) in {b"\xff\xfe", b"\xfe\xff"} else "utf-8-sig"
            raw.seek(0)
            with io.TextIOWrapper(raw, encoding=encoding, errors="replace", newline="") as stream:
                rows = csv.reader(stream, delimiter="\t" if ext == ".tsv" else ",")
                return table_preview(islice(rows, part * ROWS, None), part)
    if ext in {".docx", ".pptx"}:
        blocks = []
        length = 0
        for label, text in passages(name, path.read_bytes()):
            if not text.strip():
                continue
            length += len(text)
            if length > 100000:
                blocks.append({"label":label, "text":text[:max(0, 100000 - length + len(text))]})
                return dict(kind="document", blocks=blocks, truncated=True)
            blocks.append({"label":label, "text":text})
        return dict(kind="document", blocks=blocks)
    if ext in TEXT:
        with path.open("rb") as stream:
            data = stream.read(120004)
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        return dict(kind="text", text=data[:120000].decode(encoding, errors="replace"), truncated=len(data) > 120000)
    if ext == ".zip":
        with ZipFile(path) as archive:
            names = archive.namelist()
            return dict(kind="text", text="\n".join(names[:500]), truncated=len(names) > 500)
    return dict(kind="unavailable", reason="Preview is unavailable for this file type. Download the original to open it in a compatible app.")

def pdf_page(path, part):
    import pymupdf

    with pymupdf.open(path, filetype="pdf") as doc:
        if doc.needs_pass or part >= len(doc):
            raise ValueError("PDF page is unavailable.")
        page = doc[part]
        scale = min(2, 1650 / max(page.rect.width, page.rect.height))
        return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
