from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature = 1.5,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm=llm)

result = llm.invoke("What is the capital of India?")

print(result)