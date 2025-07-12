## Note: This code will give error while executing because open sourse LLM like Hugging Face Models cannot create structure output by default. We need to work with output parsers in order to get structured output with Open Sourse LLM.

from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.3-70B-Instruct'
    )

model = ChatHuggingFace(llm = llm)

# Schema
class Review(TypedDict):
    key_themes : Annotated[str, 'Write down all the key themes discussed in the review']
    summary: Annotated[str, 'A bried summary of the review']
    sentiment: Annotated[str, 'Return sentiment of the review either positive, negative or neutral']
    sentiment_score : Annotated[str, 'Sentiment score of the review out of 10']
    prod : Annotated[Optional[list[str]], 'Write down all the pros inside a list']
    cons : Annotated[Optional[list[str]], 'Write down all the cons inside a list']
    name : Annotated[Optional[str], 'Write the name of reviewer']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke('''
                                 
The Dell XPS 15 (2024) is a premium laptop that combines high-end performance with an elegant and durable design. It features a sleek aluminum chassis with a soft-touch carbon fiber interior, giving it a refined yet sturdy build quality. The 15.6-inch OLED 3.5K display offers rich colors, sharp details, and deep contrast, making it ideal for content creators and media consumption. Performance-wise, it is powered by Intel’s 14th Gen i7 processor and an NVIDIA RTX 4050 GPU, allowing smooth multitasking, video editing, and light gaming. The 32GB RAM and 1TB SSD ensure fast boot times and responsive application performance. Battery life is satisfactory for a performance laptop, typically lasting between seven to eight hours with light usage, though the high-resolution OLED screen can drain it faster under heavy load. The keyboard offers a comfortable typing experience with good key travel, and the trackpad is large and highly responsive. While the laptop includes Thunderbolt 4 ports, an SD card reader, and a headphone jack, the absence of USB-A ports may require carrying additional adapters.

What’s Good:
The Dell XPS 15 impresses with its vibrant OLED display, excellent build quality, and top-tier performance. Its smooth multitasking ability and sleek, modern design make it suitable for both work and entertainment.

What’s Not:
It lacks traditional USB-A ports, and battery life could be better, especially under heavy use. The price is also on the higher side, which may not suit budget-conscious buyers.
                                 
                                 
Review by Shahrukh Ahmad 
                                 
''')


for feature,desc in result.items():
    print(f'{feature} : {desc}')
