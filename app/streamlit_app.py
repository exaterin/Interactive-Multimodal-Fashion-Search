import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from components.search_ui import render_search_tab
from components.chat_ui import render_chat_tab


st.set_page_config(
    page_title="Fashion Conversational Search",
    layout="wide",
)

st.title("Fashion Conversational Search")
st.caption("Text retrieval with attribute filters and a simple chatbot interface.")

# Create two tabs
tab_search, tab_chat = st.tabs(["Search", "Chatbot"])

# Search interface
with tab_search:
    render_search_tab()

# Chatbot interface
with tab_chat:
    render_chat_tab()