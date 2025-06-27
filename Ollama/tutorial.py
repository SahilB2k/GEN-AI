from transformers import pipeline
from langchain_community.llms import ollama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Fix: Create the Hugging Face pipeline first
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.8,
    max_new_tokens=150
)

# Then pass it to LangChain
llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# Environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide a helpful response to the user's queries."),
    ("user", "Question: {question}")
])

output_parser = StrOutputParser()
chain = prompt | model | output_parser

# Streamlit
st.title("Langchain Demo")
input_text = st.text_input("Enter the topic here")

if input_text:
    with st.spinner("Generating response..."):
        result = chain.invoke({"question": input_text})
        st.write(result)
