"""Local archive for original uploads and extracted text."""
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

from .attachments import extract
from .visual import is_photo, scan_aware_pdf


class ArtifactStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name, data):
        name = Path(name.replace("\\", "/")).name or "upload"
        artifact_id = uuid4().hex
        try:
            chunks = [] if is_photo(name) else extract(name, data)
            issue = None
        except Exception as exc:
            chunks = []
            issue = str(exc) or "Text extraction failed."
        metadata = dict(id=artifact_id, name=name, size=len(data), issue=issue)
        with TemporaryDirectory(dir=self.directory) as temp:
            pending = Path(temp) / "pending"
            pending.mkdir()
            (pending / "original").write_bytes(data)
            (pending / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (pending / "passages.json").write_text(json.dumps(chunks), encoding="utf-8")
            pending.rename(self.directory / artifact_id)
        return self.metadata(artifact_id)

    def metadata(self, artifact_id):
        if not re.fullmatch(r"[a-f0-9]{32}", artifact_id):
            raise FileNotFoundError(artifact_id)
        path = self.directory / artifact_id / "metadata.json"
        entry = json.loads(path.read_text(encoding="utf-8"))
        if is_photo(entry["name"]) or (
            Path(entry["name"]).suffix.lower() == ".pdf" and
            (entry["issue"] or "").startswith("No readable text")
        ):
            entry["issue"] = None
        return entry

    def list(self):
        entries = []
        for path in self.directory.glob("*/metadata.json"):
            try:
                modified = path.stat().st_mtime
                entries.append((modified, self.metadata(path.parent.name)))
            except FileNotFoundError:
                continue
        return [entry for _, entry in sorted(entries, key=lambda item: item[0], reverse=True)]

    def remove(self, artifact_id):
        self.metadata(artifact_id)
        # Move the complete entry out of the library before removing its contents.
        with TemporaryDirectory(dir=self.directory) as staging:
            (self.directory / artifact_id).rename(Path(staging) / "removed")

    def passages(self, artifact_id):
        metadata = self.metadata(artifact_id)
        if metadata["issue"]:
            raise ValueError(f'{metadata["name"]}: {metadata["issue"]}')
        directory = self.directory / artifact_id
        if is_photo(metadata["name"]):
            return []
        if Path(metadata["name"]).suffix.lower() != ".pdf":
            return json.loads((directory / "passages.json").read_text(encoding="utf-8"))
        cache = directory / "ocr-passages.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        data = (directory / "original").read_bytes()
        try:
            chunks = extract(metadata["name"], data, pages=scan_aware_pdf(data))
        except ValueError as exc:
            if not str(exc).startswith("No readable text"):
                raise
            chunks = []
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as output:
            temporary = Path(output.name)
            json.dump(chunks, output)
        try:
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return chunks
