from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="conversational"
)

chat_model = ChatHuggingFace(llm=llm,max_new_tokens=20   )

result = chat_model.invoke("What is the capital of India?")
print(result.content)
