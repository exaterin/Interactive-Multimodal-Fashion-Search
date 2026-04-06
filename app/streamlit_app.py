import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.search_ui import render_search_tab
from components.chat_ui import render_chat_tab
from src.conversation.llm_client import LLMClient


def main() -> None:
    st.set_page_config(page_title="Fashion Search App", layout="wide")

    llm_client = LLMClient(
        model="google/gemini-3-flash-preview",
        temperature=0.0,
        timeout=60,
    )

    tab1, tab2 = st.tabs(["Search", "Chatbot"])

    with tab1:
        render_search_tab()

    with tab2:
        render_chat_tab(llm_client)


if __name__ == "__main__":
    main()