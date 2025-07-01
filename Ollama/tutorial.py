from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import streamlit as st 
import os

load_dotenv()

# Optional: if you use LangSmith or similar
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide a helpful response to the user queries."),
    ("user", "Question: {question}")
])

# 🌟 Title
st.title("🧠 Like-a-10: Smart Explainer Bot")

# 📝 Text Input
input_text = st.text_input("💬 Enter your question here:")

# 🔧 Customization Options
# st.markdown("### 🛠️ Customize the Explanation Style:")
# child_level = st.checkbox("🧒 Explain like I'm 10")
# expert_level = st.checkbox("🧠 Explain it like an expert")
# example_only = st.checkbox("📚 Explain it with examples only")

mode=st.radio("Choose Explaination style:",[
    "Default",
    "Child-Friendly",
    "Examples Only",
    "Expert"
])

# 🔮 LLM Configuration
llm = OllamaLLM(model="gemma3", model_kwargs={"num_predict": 150}, temperature=0.8)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    print("Input received:", input_text)
    st.success("✅ Input received!")

    with st.spinner("⏳ Generating your results..."):
        print("Inside spinner")

        # if child_level and expert_level:
        if mode=="Child-Friendly" and "Expert":
            st.markdown("<h3>🤔 Are you an Expert or a Child? Choose one!</h3>", unsafe_allow_html=True)

        # elif child_level:
        elif mode=="Child-Friendly":
            child_prompt = f"Explain '{input_text}' like I am 10 years old using real-life examples"
            response = chain.invoke({"question": child_prompt})
            st.subheader("🧒 Explaining to a 10yr old...")
            print("Calling LLM...")
            st.write(response)

        # elif example_only:
        elif mode=="Examples Only":
            example_prompt = f"Explain '{input_text}' with examples only!"
            response = chain.invoke({"question": example_prompt})
            st.subheader("📚 Explaining using examples...")
            print("Calling LLM...")
            st.write(response)

        # elif expert_level:
        elif mode=="Expert":
            expert_prompt = f"Explain '{input_text}' using technical and complex terminology"
            response = chain.invoke({"question": expert_prompt})
            st.subheader("🧠 Explaining to an Expert...")
            print("Calling LLM...")
            st.write(response)

        else :
            response = chain.invoke({"question": input_text})
            st.subheader("💡 Standard Explanation")
            st.write(response)
