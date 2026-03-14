# utils/data_loader.py
from __future__ import annotations

import io
import logging
import os
import ssl
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration SSL pour le téléchargement des modèles OCR
try:
    import certifi

    CERT_FILE = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", CERT_FILE)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", CERT_FILE)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=CERT_FILE)
except Exception as e:
    logging.warning(f"Impossible d'initialiser certifi pour SSL: {e}")
    CERT_FILE = None

# Dépendances OCR optionnelles
try:
    import fitz  # PyMuPDF
except ImportError as e:
    logging.warning(f"PyMuPDF non disponible: {e}")
    fitz = None

try:
    from PIL import Image
except ImportError as e:
    logging.warning(f"Pillow non disponible: {e}")
    Image = None

_easyocr_reader = None
_easyocr_failed = False


# --- Fonctions OCR ---
def get_easyocr_reader():
    """Initialise EasyOCR à la demande, une seule fois."""
    global _easyocr_reader, _easyocr_failed

    if _easyocr_reader is not None:
        return _easyocr_reader
    if _easyocr_failed:
        return None

    try:
        import easyocr

        logging.info("Initialisation du lecteur EasyOCR...")
        _easyocr_reader = easyocr.Reader(["en", "fr"], gpu=False)
        logging.info("Lecteur EasyOCR initialisé.")
        return _easyocr_reader
    except ImportError as e:
        logging.warning(
            f"EasyOCR non installé: {e}. L'OCR pour PDF ne sera pas disponible."
        )
    except Exception as e:
        logging.error(f"Erreur inattendue lors du chargement des modules/modèle OCR: {e}")

    _easyocr_failed = True
    return None


def extract_text_from_pdf_with_ocr(file_path: str) -> Optional[str]:
    """Extrait le texte d'un PDF en utilisant EasyOCR."""
    reader = get_easyocr_reader()
    if fitz is None or Image is None or reader is None:
        logging.warning("Modules/Modèle OCR non disponibles. Impossible d'effectuer l'OCR.")
        return None

    doc = None
    text_content: list[str] = []

    try:
        doc = fitz.open(file_path)
        for page_num in tqdm(range(len(doc)), desc=f"OCR de {os.path.basename(file_path)}"):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                img_np = np.array(img)
                results = reader.readtext(img_np)
                page_text = "\n".join(
                    res[1].strip() for res in results if len(res) > 1 and str(res[1]).strip()
                )
                if page_text:
                    text_content.append(page_text)
            except Exception as ocr_e:
                logging.error(
                    f"Erreur lors de l'OCR de la page {page_num + 1} de {file_path}: {ocr_e}"
                )
                continue

        full_text = "\n".join(text_content).strip()
        if full_text:
            logging.info(f"Texte extrait via OCR de PDF: {file_path} ({len(full_text)} caractères)")
            return full_text

        logging.warning(f"Aucun texte significatif extrait via OCR de {file_path}.")
        return None
    except Exception as e:
        logging.error(f"Erreur lors de l'ouverture ou du traitement OCR du PDF {file_path}: {e}")
        return None
    finally:
        if doc is not None:
            doc.close()


# --- Fonctions d'extraction de texte ---
def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extrait le texte d'un PDF, avec fallback OCR si peu de texte est trouvé."""
    try:
        from PyPDF2 import PdfReader

        pdf_reader = PdfReader(file_path)
        text_parts: list[str] = []

        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
            except Exception:
                page_text = None
            if page_text:
                text_parts.append(page_text + "\n")

        text = "".join(text_parts).strip()

        if len(text) < 100:
            logging.info(
                f"Peu de texte trouvé dans {file_path} via extraction standard ({len(text)} caractères). Tentative d'OCR..."
            )
            ocr_text = extract_text_from_pdf_with_ocr(file_path)
            if ocr_text:
                return ocr_text

            logging.warning(f"L'OCR n'a pas non plus produit de texte significatif pour {file_path}.")
            return text or None

        logging.info(f"Texte extrait de PDF: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {file_path}: {e}. Tentative d'OCR en dernier recours...")
        ocr_text = extract_text_from_pdf_with_ocr(file_path)
        if ocr_text:
            return ocr_text

        logging.warning(
            f"L'OCR n'a pas non plus produit de texte significatif après échec de l'extraction standard pour {file_path}."
        )
        return None


def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx

        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None


def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None


def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd

        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
        except Exception as read_e:
            logging.warning(
                f"Erreur lecture CSV {file_path}: {read_e}. Tentative avec séparateur ';'"
            )
            try:
                df = pd.read_csv(file_path, sep=";")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep=";", encoding="latin1")
            except Exception as read_e2:
                logging.error(f"Impossible de lire le CSV {file_path}: {read_e2}")
                return None

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except ImportError:
        logging.warning("Pandas non installé. Impossible de lire les fichiers CSV.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None


def extract_text_from_excel(file_path: str) -> Optional[Union[str, Dict[str, str]]]:
    """Extrait le texte de chaque feuille d'un fichier Excel."""
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        sheets_data: dict[str, str] = {}
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            sheets_data[sheet_name] = df.to_string()

        logging.info(f"Texte extrait de {len(sheets_data)} feuille(s) dans Excel: {file_path}")
        if len(sheets_data) == 1:
            return next(iter(sheets_data.values()))
        return sheets_data
    except ImportError:
        logging.warning("Pandas ou openpyxl non installé. Impossible de lire les fichiers Excel.")
        return None
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None


# --- Fonctions de chargement ---
def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un fichier ZIP depuis une URL et l'extrait."""
    if not url:
        logging.warning("Aucune URL fournie pour le téléchargement.")
        return False

    try:
        logging.info(f"Téléchargement des données depuis {url}...")
        request_kwargs = {"timeout": 60}
        if CERT_FILE:
            request_kwargs["verify"] = CERT_FILE

        response = requests.get(url, **request_kwargs)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extraction du contenu dans {output_dir}...")
            z.extractall(output_dir)

        logging.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de téléchargement: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("Le fichier téléchargé n'est pas un ZIP valide.")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement/extraction: {e}")
        return False


def load_and_parse_files(input_dir: str) -> List[Dict[str, Any]]:
    """
    Charge et parse récursivement les fichiers d'un répertoire.
    Retourne une liste de dictionnaires, chacun représentant un document.
    """
    documents: list[dict[str, Any]] = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return []

    logging.info(f"Parcours du répertoire source: {input_dir}")
    for file_path in input_path.rglob("*.*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(input_path)
        source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        ext = file_path.suffix.lower()

        logging.debug(f"Traitement du fichier: {relative_path} (Dossier source: {source_folder})")

        extracted_content = None
        if ext == ".pdf":
            extracted_content = extract_text_from_pdf(str(file_path))
        elif ext == ".docx":
            extracted_content = extract_text_from_docx(str(file_path))
        elif ext == ".txt":
            extracted_content = extract_text_from_txt(str(file_path))
        elif ext == ".csv":
            extracted_content = extract_text_from_csv(str(file_path))
        elif ext in [".xlsx", ".xls"]:
            extracted_content = extract_text_from_excel(str(file_path))
        else:
            logging.warning(f"Type de fichier non supporté ignoré: {relative_path}")
            continue

        if not extracted_content:
            logging.warning(f"Aucun contenu n'a pu être extrait de {relative_path}")
            continue

        if isinstance(extracted_content, dict):
            for sheet_name, text in extracted_content.items():
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
                    "page_content": extracted_content,
                    "metadata": {
                        "source": str(relative_path),
                        "filename": file_path.name,
                        "category": source_folder,
                        "full_path": str(file_path.resolve()),
                    },
                }
            )

    logging.info(f"{len(documents)} documents chargés et parsés.")
    return documents