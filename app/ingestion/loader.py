"""Load supported local files into plain-text documents."""

from pathlib import Path

from app.models import Document


def load_file(path: str | Path, metadata: dict | None = None) -> Document:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires `pip install pypdf`.") from exc
        text = "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    else:
        raise ValueError(f"Unsupported document type: {suffix or '(no extension)'}")

    document_metadata = {"source": str(file_path)}
    document_metadata.update(metadata or {})
    return Document(text=text, metadata=document_metadata)