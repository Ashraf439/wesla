"""Local OCR and image preparation for uploaded files."""
import base64
from functools import lru_cache
import io
import re
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from pathlib import Path

from .activity import checkpoint
from .log import get_logger

log = get_logger(__name__)

PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif", ".tif", ".tiff"}
OCR_PAGE_LIMIT = 12
OCR_PAGE_TIMEOUT_S = 25
# Total OCR time per PDF. Requests hold the server lock while a scan is read, so a slow
# OCR engine must not block everyone for 12 pages x 25 s.
OCR_PDF_BUDGET_S = 90

def is_photo(name):
    return Path(name).suffix.lower() in PHOTO_EXTENSIONS

@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)

def _read_text_windows(png):
    script = Path(__file__).with_name("ocr_windows.ps1")
    with TemporaryDirectory() as folder:
        image = Path(folder) / "page.png"
        image.write_bytes(png)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             script.read_text(encoding="utf-8")],
            env={**os.environ, "WESLA_OCR_IMAGE": str(image)},
            capture_output=True, encoding="utf-8", errors="replace", timeout=OCR_PAGE_TIMEOUT_S,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        # Vision can still interpret the image if Windows has no installed OCR language.
        log.warning("Windows OCR exited with code %s: %s", result.returncode, (result.stderr or "").strip()[:300])
        return ""

def read_text(png):
    """Best-effort OCR. Returns "" instead of raising, so one unreadable or slow page never
    fails the whole request (and the PDF's OCR cache still gets written)."""
    try:
        if sys.platform == "win32":
            return _read_text_windows(png)
        rows, _ = ocr_engine()(png)
        return "\n".join(row[1] for row in (rows or []))
    except subprocess.TimeoutExpired:
        log.warning("OCR gave up on an image after %ss", OCR_PAGE_TIMEOUT_S)
    except Exception as exc:  # missing OCR engine, PowerShell unavailable, corrupt image...
        log.warning("OCR failed: %s: %s", type(exc).__name__, exc)
    return ""

def png_image(data, size=1050):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Image exceeds 40 megapixels. Resize it before analysis.")
        image = ImageOps.exif_transpose(source)
        image.thumbnail((size, size))
        background = Image.new("RGB", image.size, "white")
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background.paste(rgba, mask=rgba.getchannel("A"))
        else:
            background.paste(image.convert("RGB"))
        output = io.BytesIO()
        background.save(output, format="PNG")
        return output.getvalue()

def encode_image(data):
    return base64.b64encode(data).decode("ascii")

def page_png(page, size):
    import pymupdf

    scale = min(3, size / max(page.rect.width, page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")

def scan_aware_pdf(data):
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("Unlock this PDF before analyzing it.")
        scanned = 0
        started = monotonic()
        budget_logged = False
        for page in doc:
            checkpoint()
            label = f"page {page.number + 1}"
            text = page.get_text(sort=True)
            char_count = sum(c.isalnum() for c in text)
            page_area = page.rect.get_area()
            has_large_image = any(
                pymupdf.Rect(image["bbox"]).get_area() >= page_area * 0.35
                for image in page.get_image_info()
            ) if char_count < 1800 else False
            needs_ocr = scanned < OCR_PAGE_LIMIT and (char_count < 200 or has_large_image)
            if needs_ocr and monotonic() - started > OCR_PDF_BUDGET_S:
                if not budget_logged:
                    log.warning("OCR time budget (%ss) used up at page %d; remaining pages use their "
                                "embedded text only", OCR_PDF_BUDGET_S, page.number + 1)
                    budget_logged = True
                needs_ocr = False
            if needs_ocr:
                checkpoint("Reading scanned PDF page " + str(page.number + 1))
                scanned += 1
                recognized = read_text(page_png(page, 1750))
                if sum(c.isalnum() for c in recognized) > char_count:
                    text, label = recognized, label + " (OCR)"
            yield label, text

def pdf_images(data, names, scan=()):
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        scanned = next((re.search(r" · page (\d+) \(OCR\)$", chunk.source)
                        for chunk in scan if chunk.source.endswith("(OCR)")), None)
        page = doc[int(scanned[1]) - 1 if scanned else 0]
        return [(f"{names} · page {page.number + 1}", encode_image(page_png(page, 1050)))]
