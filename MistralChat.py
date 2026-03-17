"""Interface Streamlit minimale pour le service RAG + SQL."""

from __future__ import annotations

import logging

import streamlit as st

from api import get_rag_service
from utils.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()


@st.cache_resource
def _get_service():
    return get_rag_service()


st.title(SETTINGS.app_title)
st.caption(f"Assistant virtuel pour {SETTINGS.app_name}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Bonjour. Je suis votre assistant {SETTINGS.app_name}. "
                "Posez une question et je répondrai avec le pipeline RAG + SQL."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input(f"Posez votre question sur la {SETTINGS.app_name}...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.text("...")
        try:
            result = _get_service().ask(question, SETTINGS.search_k)
            placeholder.write(result.answer)
            with st.expander("Détails techniques"):
                st.write(
                    {
                        "retrieval_count": result.retrieval_count,
                        "sql_status": result.sql_status,
                        "sql_query": result.sql_query,
                        "latency_total_s": result.latency_total_s,
                    }
                )
        except Exception as exc:
            LOGGER.exception("Erreur Streamlit pendant ask")
            placeholder.write(f"Erreur: {exc}")
            result = None

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer if result else "Erreur technique."}
    )
