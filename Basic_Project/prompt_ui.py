from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational"
)

model = ChatHuggingFace(llm=llm,max_new_tokens=150)

st.header("Research Tool")
paper_input=st.selectbox("Select Reserach Paper Name",["Attention is All ou Need","BERT:training of Deep Bidirectional Transformers","GPT-3:Language Models are Few-Shot Learners","Diffusion Models Beat GANs on Image Sysnthesis"])

style_input=st.selectbox("Select Explaination Style",["Beginner-Friendly","Technical","Code-Oriented","Mathematical"])

length_input=st.selectbox("Select Explaination Length",["Short(1-2 paragraphs)","Medium(3-5 paragraphs)","Long(detailed explaination)"])

# template can also be created using f String but promptTemplate give various benefits like - default validation , template can be converted in JSON format so that it can be reused
template=PromptTemplate(
    template= """
    Please summarize the reserach paper titled "{paper_input}" with the following spcification:
    Explaination Style:{style_input}
    Explaination length:{length_input}
    1.Mathematical Details:
        -Include relevant mathematical equations if present in the paper.
        -Explain the mathematical conecpts using simple,intutive code snippets where applicable.
    2.Analogies:
        -use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper,repond with:"insufficient information available" instead of guessing.
    Ensure the summary is clear, accurate and aligned with the provided style and length. """,
    
    input_variables=['paper_input','style_input','length_input']
)

# NORMAL USECASE
# prompt=template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })

# if st.button("Summarize"):
#     result = model.invoke(prompt)
#     st.write(result.content)

# USING CHAIN - helps reduce code size 
if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input}
    )
    st.write(result.content)

  