from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=1.8,
        max_new_tokens=150
    )
)

model=ChatHuggingFace(llm=llm)
Result=model.invoke("summarize Gravity for a 10 year old using examples")
print(Result.content)