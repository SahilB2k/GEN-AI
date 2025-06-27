from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3")  # or "mistral", etc.
print("Calling LLM...")
res = llm.invoke("What is gravity?")
print("Response:", res)
