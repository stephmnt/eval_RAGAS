"""Script minimal d'indexation RAG."""

from __future__ import annotations

import argparse
import logging

from utils.config import get_settings
from utils.data_loader import load_and_parse_files
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()


def run_indexing(input_directory: str) -> None:
    LOGGER.info("Démarrage indexation depuis: %s", input_directory)
    documents = load_and_parse_files(input_directory)
    if not documents:
        LOGGER.warning("Aucun document exploitable. Indexation interrompue.")
        return

    vector_store = VectorStoreManager()
    vector_store.build_index(documents)

    LOGGER.info("Indexation terminée. Documents traités: %s", len(documents))
    if vector_store.index is not None:
        LOGGER.info("Chunks indexés: %s", vector_store.index.ntotal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation des données pour le système RAG.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=SETTINGS.input_dir,
        help=f"Répertoire source (par défaut: {SETTINGS.input_dir})",
    )
    args = parser.parse_args()
    run_indexing(args.input_dir)
