from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral

from .config import get_settings

SETTINGS = get_settings()
LOGGER = logging.getLogger(__name__)


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.document_chunks: list[dict[str, Any]] = []
        self.mistral_client = Mistral(api_key=SETTINGS.mistral_api_key)
        self._index_path = Path(SETTINGS.faiss_index_file)
        self._chunks_path = Path(SETTINGS.document_chunks_file)
        self._load_index_and_chunks()

    def _reset_state(self, *, remove_files: bool = False) -> None:
        self.index = None
        self.document_chunks = []
        if remove_files:
            self._index_path.unlink(missing_ok=True)
            self._chunks_path.unlink(missing_ok=True)

    def _load_index_and_chunks(self) -> None:
        if not self._index_path.exists() or not self._chunks_path.exists():
            LOGGER.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")
            return

        try:
            LOGGER.info("Chargement de l'index Faiss depuis %s...", self._index_path)
            self.index = faiss.read_index(str(self._index_path))
            LOGGER.info("Chargement des chunks depuis %s...", self._chunks_path)
            with self._chunks_path.open("rb") as handle:
                self.document_chunks = pickle.load(handle)
            LOGGER.info(
                "Index (%s vecteurs) et %s chunks chargés.",
                self.index.ntotal,
                len(self.document_chunks),
            )
        except Exception as exc:
            LOGGER.error("Erreur lors du chargement de l'index/chunks: %s", exc)
            self._reset_state()

    def _split_documents_to_chunks(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        LOGGER.info(
            "Découpage de %s documents en chunks (taille=%s, chevauchement=%s)...",
            len(documents),
            SETTINGS.chunk_size,
            SETTINGS.chunk_overlap,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SETTINGS.chunk_size,
            chunk_overlap=SETTINGS.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        all_chunks: list[dict[str, Any]] = []
        for doc_index, doc in enumerate(documents):
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            LOGGER.info(
                "Document '%s' découpé en %s chunks.",
                doc["metadata"].get("filename", "N/A"),
                len(chunks),
            )
            all_chunks.extend(
                {
                    "id": f"{doc_index}_{chunk_index}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": chunk_index,
                        "start_index": chunk.metadata.get("start_index", -1),
                    },
                }
                for chunk_index, chunk in enumerate(chunks)
            )

        LOGGER.info("Total de %s chunks créés.", len(all_chunks))
        return all_chunks

    def _generate_embeddings(self, chunks: list[dict[str, Any]]) -> np.ndarray | None:
        if not SETTINGS.mistral_api_key:
            LOGGER.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            LOGGER.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        LOGGER.info(
            "Génération des embeddings pour %s chunks (modèle: %s)...",
            len(chunks),
            SETTINGS.embedding_model,
        )
        all_embeddings: list[list[float]] = []
        embedding_dim: int | None = None
        total_batches = (len(chunks) + SETTINGS.embedding_batch_size - 1) // SETTINGS.embedding_batch_size

        for start in range(0, len(chunks), SETTINGS.embedding_batch_size):
            batch_num = (start // SETTINGS.embedding_batch_size) + 1
            batch_chunks = chunks[start : start + SETTINGS.embedding_batch_size]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            LOGGER.info(
                "Traitement du lot %s/%s (%s chunks)",
                batch_num,
                total_batches,
                len(texts_to_embed),
            )
            try:
                response = self.mistral_client.embeddings.create(
                    model=SETTINGS.embedding_model,
                    inputs=texts_to_embed,
                )
                batch_embeddings = [data.embedding for data in response.data]
                if batch_embeddings and embedding_dim is None:
                    embedding_dim = len(batch_embeddings[0])
            except Exception as exc:
                LOGGER.error(
                    "Erreur API Mistral lors de la génération d'embeddings (lot %s): %s",
                    batch_num,
                    exc,
                )
                if embedding_dim is None:
                    LOGGER.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                    continue
                LOGGER.warning(
                    "Ajout de %s vecteurs nuls de dimension %s pour le lot échoué.",
                    len(texts_to_embed),
                    embedding_dim,
                )
                batch_embeddings = [np.zeros(embedding_dim, dtype="float32").tolist() for _ in texts_to_embed]
            all_embeddings.extend(batch_embeddings)

        if not all_embeddings:
            LOGGER.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.asarray(all_embeddings, dtype="float32")
        LOGGER.info("Embeddings générés avec succès. Shape: %s", embeddings_array.shape)
        return embeddings_array

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            LOGGER.warning("Aucun document fourni pour construire l'index.")
            return

        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            LOGGER.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            LOGGER.error(
                "Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks."
            )
            self._reset_state(remove_files=True)
            return

        dimension = embeddings.shape[1]
        LOGGER.info(
            "Création de l'index Faiss optimisé pour la similarité cosinus avec dimension %s...",
            dimension,
        )
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        LOGGER.info("Index Faiss créé avec %s vecteurs.", self.index.ntotal)
        self._save_index_and_chunks()

    def _save_index_and_chunks(self) -> None:
        if self.index is None or not self.document_chunks:
            LOGGER.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            LOGGER.info("Sauvegarde de l'index Faiss dans %s...", self._index_path)
            faiss.write_index(self.index, str(self._index_path))
            LOGGER.info("Sauvegarde des chunks dans %s...", self._chunks_path)
            with self._chunks_path.open("wb") as handle:
                pickle.dump(self.document_chunks, handle)
            LOGGER.info("Index et chunks sauvegardés avec succès.")
        except Exception as exc:
            LOGGER.error("Erreur lors de la sauvegarde de l'index/chunks: %s", exc)

    def search(self, query_text: str, k: int = 5, min_score: float | None = None) -> list[dict[str, Any]]:
        if self.index is None or not self.document_chunks:
            LOGGER.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not SETTINGS.mistral_api_key:
            LOGGER.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
            return []

        LOGGER.info("Recherche des %s chunks les plus pertinents pour: '%s'", k, query_text)
        try:
            response = self.mistral_client.embeddings.create(
                model=SETTINGS.embedding_model,
                inputs=[query_text],
            )
            query_embedding = np.asarray([response.data[0].embedding], dtype="float32")

            faiss.normalize_L2(query_embedding)
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            results: list[dict[str, Any]] = []
            min_score_percent = min_score * 100 if min_score is not None else None
            for rank, chunk_index in enumerate(indices[0] if indices.size > 0 else []):
                if not 0 <= chunk_index < len(self.document_chunks):
                    LOGGER.warning(
                        "Index Faiss %s hors limites (taille des chunks: %s).",
                        chunk_index,
                        len(self.document_chunks),
                    )
                    continue

                raw_score = float(scores[0][rank])
                similarity = raw_score * 100
                if min_score_percent is not None and similarity < min_score_percent:
                    LOGGER.debug(
                        "Document filtré (score %.2f%% < minimum %.2f%%)",
                        similarity,
                        min_score_percent,
                    )
                    continue

                chunk = self.document_chunks[chunk_index]
                results.append(
                    {
                        "score": similarity,
                        "raw_score": raw_score,
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:k]

            if min_score_percent is not None:
                LOGGER.info("%s chunks pertinents trouvés (score minimum: %.2f%%).", len(results), min_score_percent)
            else:
                LOGGER.info("%s chunks pertinents trouvés.", len(results))

            return results

        except Exception as exc:
            LOGGER.error("Erreur inattendue lors de la recherche: %s", exc)
            return []
