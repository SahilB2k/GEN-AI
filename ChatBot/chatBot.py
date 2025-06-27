from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=1.5,
        max_new_tokens=150
    )

        
)

model=ChatHuggingFace(llm=llm)

while True:
    user_input=input()