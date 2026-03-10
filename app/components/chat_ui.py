import streamlit as st
from src.conversation.chat_manager import ChatManager


def init_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def clear_chat():
    st.session_state.chat_messages = []


def render_chat_history():
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_chat_tab(chat_manager: ChatManager):

    init_chat_state()

    st.title("Chatbot")

    if st.button("Clear chat"):
        clear_chat()
        st.rerun()

    # Chat history container
    chat_container = st.container()

    with chat_container:
        render_chat_history()

    # Chat input at the bottom
    user_input = st.chat_input("Write a message...")

    if user_input:

        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_messages.append(user_message)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = chat_manager.generate_reply(
                            st.session_state.chat_messages
                        )
                    except Exception as e:
                        reply = f"Error: {e}"

                    st.markdown(reply)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply}
        )