__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import uuid
from datetime import datetime
from bot import create_application_logs, insert_application_logs, get_chat_history, rag_chain, MODEL

# Initialize database and tables
create_application_logs()

# Generate a unique session ID for each user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# Streamlit UI
st.title("Personal Finance Mentor")
st.sidebar.header("Chat History")
st.sidebar.write("Conversation history will appear here.")

# User Input
question = st.text_input("Ask a financial question:", "")

if st.button("Ask") and question.strip():
    # Fetch chat history
    chat_history = get_chat_history(session_id)

    # Get AI response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']

    # Save to logs
    insert_application_logs(session_id, question, response, MODEL)

    # Display response
    st.markdown(f"**You:** {question}")
    st.markdown(f"**AI:** {response}")

    # Update sidebar with chat history
    chat_history = get_chat_history(session_id)
    with st.sidebar:
        for msg in chat_history:
            role = "You" if msg["role"] == "human" else "AI"
            st.write(f"**{role}:** {msg['content']}")

# View Chat History
if st.sidebar.button("Clear Chat History"):
    st.session_state.session_id = str(uuid.uuid4())  # Reset session
    st.sidebar.write("Chat history cleared.")
