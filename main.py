import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Streamlit page setup
st.set_page_config(page_title="🧠 LangChain LLM Chat with Memory")
st.title("💬 Chat with Memory via OpenRouter")

# Session state memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Load LLM from OpenRouter
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct:free",
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1"
)

# Build conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

# Get user input
user_input = st.text_input("Ask something:")

if user_input:
    with st.spinner("Thinking..."):
        response = conversation.predict(input=user_input)
        st.session_state.last_user_input = user_input
        st.session_state.last_response = response

# Display past conversation
if st.session_state.get("memory"):
    messages = st.session_state.memory.chat_memory.messages
    for msg in messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")
