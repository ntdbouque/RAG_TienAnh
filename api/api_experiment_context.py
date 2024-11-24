import gradio as gr

import random

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI


import torch

global selected_chunk
global selected_document

import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/competitions/Sly/PaperRetrieval/RAG_tool/cache_transformer_modelse"

### CONSTANT:
TEXT_SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=64)
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
CONTEXTUAL_PROMPT = """<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""




def load_document(document_path):
    documents = SimpleDirectoryReader(input_files=[document_path]).load_data(num_workers=9, show_progress=True)
    return documents

def split_document(selectived_document):
    process_documents = TEXT_SPLITTER(selectived_document)
    return process_documents


def parse_file_and_chunking(document_path):
    global selected_chunk, selected_document
    selected_document = load_document(document_path=document_path)
    selected_chunk = split_document(selected_document)
    print('complete')
    gr.Warning('Parse and Split Successfully')


def generate_text(document_path, model_name, temperature, top_k, top_p):
    global selected_document, selected_chunk
    
    rand_idx_chunk = random.randint(0, len(selected_chunk)-1)
    rand_idx_doc = random.randint(0, len(selected_document)-1)
    
    prompt = CONTEXTUAL_PROMPT.format(WHOLE_DOCUMENT=selected_document[rand_idx_doc].text, CHUNK_CONTENT=selected_chunk[rand_idx_chunk].text)
    #prompt = 'Hey, are you conscious? Can you talk to me?'
    messages = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
                ChatMessage(
                    role="user",
                    content=prompt
                ),
            ]
    llm = OpenAI(model='gpt-3.5-turbo', api_key='API-KEY')
    response = llm.chat(messages)
    contextualized_content = response.message.content
    return contextualized_content

### MAIN FUNCTION:
def process_chunk(document_path, model_name, temperature, top_k, top_p):
    context = generate_text(document_path, model_name, temperature, top_k, top_p)
    return context

# UI:
with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column(scale=1):  
            with gr.Group():
                model_name = gr.Textbox(label='Chỉnh Model', value='meta-llama/Llama-3.2-1B')
                temperature = gr.Slider(label="Chỉnh Temperature", minimum=0.0, maximum=1.5, value=0.67)
                top_k = gr.Number(label="Chỉnh Top K", value=60)
                top_p = gr.Slider(label="Chỉnh Top P", minimum=0.0, maximum=1.0, value=0.9)
                
                document_path = gr.Textbox(label='Chỉnh đường dẫn đến file pdf', value='/dataset/SLY/PaperRetrieval/Data_for_project/1.COMPUTER_VISION/ORS-001.pdf')
                
                ingest_btn = gr.Button('Ingest')
                
        with gr.Column(scale=3):  # Context Display
            context_display = gr.Textbox(label="Hiển thị Context", lines=20, interactive=False)

    with gr.Row():
        submit_btn = gr.Button("Submit")

    ingest_btn.click(
        parse_file_and_chunking,
        inputs=[document_path]
    )

    # Event bindings
    submit_btn.click(
        process_chunk,
        inputs=[document_path, model_name, temperature, top_k, top_p],
        outputs=[context_display]
    )
if __name__ == '__main__':
    ui.launch(share=True)
