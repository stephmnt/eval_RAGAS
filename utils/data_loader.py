"""Chargement des données source pour indexation (PDF + Excel)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

_easyocr_reader = None
_easyocr_failed = False


def _get_easyocr_reader():
    global _easyocr_reader, _easyocr_failed
    if _easyocr_reader is not None:
        return _easyocr_reader
    if _easyocr_failed:
        return None
    try:
        import easyocr

        LOGGER.info("Initialisation EasyOCR...")
        _easyocr_reader = easyocr.Reader(["en", "fr"], gpu=False)
        return _easyocr_reader
    except Exception as exc:
        LOGGER.warning("EasyOCR indisponible: %s", exc)
        _easyocr_failed = True
        return None


def _extract_text_from_pdf_with_ocr(file_path: str) -> str | None:
    reader = _get_easyocr_reader()
    if fitz is None or Image is None or reader is None:
        return None

    text_content: list[str] = []
    doc = None
    try:
        doc = fitz.open(file_path)
        for page_num in tqdm(range(len(doc)), desc=f"OCR {os.path.basename(file_path)}"):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            results = reader.readtext(img_np)
            page_text = "\n".join(
                item[1].strip() for item in results if len(item) > 1 and str(item[1]).strip()
            )
            if page_text:
                text_content.append(page_text)
        full_text = "\n".join(text_content).strip()
        return full_text or None
    except Exception as exc:
        LOGGER.error("Erreur OCR PDF %s: %s", file_path, exc)
        return None
    finally:
        if doc is not None:
            doc.close()


def extract_text_from_pdf(file_path: str) -> str | None:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if len(text) >= 100:
            return text
        LOGGER.info("Peu de texte extrait sur %s, fallback OCR...", file_path)
        return _extract_text_from_pdf_with_ocr(file_path) or (text or None)
    except Exception as exc:
        LOGGER.warning("Extraction PDF standard échouée sur %s: %s", file_path, exc)
        return _extract_text_from_pdf_with_ocr(file_path)


def _extract_text_from_excel(file_path: str) -> str | dict[str, str] | None:
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        sheets_data: dict[str, str] = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()
        if len(sheets_data) == 1:
            return next(iter(sheets_data.values()))
        return sheets_data
    except Exception as exc:
        LOGGER.error("Erreur extraction Excel %s: %s", file_path, exc)
        return None


def load_and_parse_files(input_dir: str) -> list[dict[str, Any]]:
    """Charge récursivement les PDF/Excel d'un dossier source."""
    documents: list[dict[str, Any]] = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        LOGGER.error("Le répertoire d'entrée '%s' n'existe pas.", input_dir)
        return []

    LOGGER.info("Parcours des sources: %s", input_dir)
    for file_path in input_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        relative_path = file_path.relative_to(input_path)
        source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

        if ext == ".pdf":
            extracted = extract_text_from_pdf(str(file_path))
        elif ext in {".xlsx", ".xls"}:
            extracted = _extract_text_from_excel(str(file_path))
        else:
            continue

        if not extracted:
            LOGGER.warning("Aucun contenu extrait de %s", relative_path)
            continue

        if isinstance(extracted, dict):
            for sheet_name, text in extracted.items():
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": f"{relative_path} (Feuille: {sheet_name})",
                            "filename": file_path.name,
                            "sheet": sheet_name,
                            "category": source_folder,
                            "full_path": str(file_path.resolve()),
                        },
                    }
                )
        else:
            documents.append(
                {
                    "page_content": extracted,
                    "metadata": {
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": source_folder,
                        "full_path": str(file_path.resolve()),
                    },
                }
            )

    LOGGER.info("%s document(s) chargé(s).", len(documents))
    return documents
