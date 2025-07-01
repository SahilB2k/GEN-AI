from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain

llm=OllamaLLM(model="gemma3")
memory=ConversationBufferMemory()

convo=ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

print(convo.predict(input="Hi,i am sahil"))
print(convo.predict(input="What is my name?"))

print("Conversational memory:",memory.buffer)
