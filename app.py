import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import os

st.set_page_config(page_title="QnA Agent", page_icon="🤖")
st.title("🤖 QnA Chatbot with Search & Memory")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
search = SerpAPIWrapper()
tools = [search.run]

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt="You are an amazing AI agent and can search Google for any question to give accurate, up-to-date answers."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Apna sawaal likho...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "1"}}
    )

    answer = response["messages"][-1].content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
