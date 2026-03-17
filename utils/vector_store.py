import os
import pickle
import logging
from typing import Any, Dict, List, Optional

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

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []
        self.mistral_client = Mistral(api_key=SETTINGS.mistral_api_key)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(SETTINGS.faiss_index_file) and os.path.exists(SETTINGS.document_chunks_file):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {SETTINGS.faiss_index_file}...")
                self.index = faiss.read_index(SETTINGS.faiss_index_file)
                logging.info(f"Chargement des chunks depuis {SETTINGS.document_chunks_file}...")
                with open(SETTINGS.document_chunks_file, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Découpe les documents en chunks avec métadonnées."""
        logging.info(
            f"Découpage de {len(documents)} documents en chunks "
            f"(taille={SETTINGS.chunk_size}, chevauchement={SETTINGS.chunk_overlap})..."
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SETTINGS.chunk_size,
            chunk_overlap=SETTINGS.chunk_overlap,
            length_function=len, # Important: mesure en caractères
            add_start_index=True, # Ajoute la position de début du chunk dans le document original
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            # Convertit notre format de document en format Langchain Document pour le splitter
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' découpé en {len(chunks)} chunks.")

            # Enrichit chaque chunk avec des métadonnées supplémentaires
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_counter}_{i}", # Identifiant unique du chunk (doc_index_chunk_index)
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata, # Métadonnées héritées du document (source, category, etc.)
                        "chunk_id_in_doc": i, # Position du chunk dans son document d'origine
                        "start_index": chunk.metadata.get("start_index", -1) # Position de début (en caractères)
                    }
                }
                all_chunks.append(chunk_dict)
            doc_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not SETTINGS.mistral_api_key:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(
            f"Génération des embeddings pour {len(chunks)} chunks (modèle: {SETTINGS.embedding_model})..."
        )
        all_embeddings = []
        total_batches = (len(chunks) + SETTINGS.embedding_batch_size - 1) // SETTINGS.embedding_batch_size

        for i in range(0, len(chunks), SETTINGS.embedding_batch_size):
            batch_num = (i // SETTINGS.embedding_batch_size) + 1
            batch_chunks = chunks[i:i + SETTINGS.embedding_batch_size]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings.create(
                    model=SETTINGS.embedding_model,
                    inputs=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                    continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


        if not all_embeddings:
             logging.error("Aucun embedding n'a pu être généré.")
             return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, Any]]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            # Nettoyer pour éviter un état incohérent
            self.document_chunks = []
            self.index = None
            # Supprimer les fichiers potentiellement corrompus
            if os.path.exists(SETTINGS.faiss_index_file): os.remove(SETTINGS.faiss_index_file)
            if os.path.exists(SETTINGS.document_chunks_file): os.remove(SETTINGS.document_chunks_file)
            return


        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Sauvegarder l'index et les chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(SETTINGS.faiss_index_file), exist_ok=True)
        os.makedirs(os.path.dirname(SETTINGS.document_chunks_file), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {SETTINGS.faiss_index_file}...")
            faiss.write_index(self.index, SETTINGS.faiss_index_file)
            logging.info(f"Sauvegarde des chunks dans {SETTINGS.document_chunks_file}...")
            with open(SETTINGS.document_chunks_file, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float | None = None) -> List[Dict[str, Any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not SETTINGS.mistral_api_key:
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=SETTINGS.embedding_model,
                inputs=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.document_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0: # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks): # Vérifier la validité de l'index
                        chunk = self.document_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []
