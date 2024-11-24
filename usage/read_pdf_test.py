import glob
import os
from llama_index.core import SimpleDirectoryReader

content_log_path = '/workspace/competitions/Sly/PaperRetrieval/RAG_tool/usage/content_log'
cv_dir = '/dataset/SLY/PaperRetrieval/test'



if __name__ == '__main__':
    documents = SimpleDirectoryReader(input_dir=cv_dir).load_data(num_workers=9, show_progress=True)
    
    count = 0
    
    for document in documents:
        content = document.text
        with open(os.path.join(content_log_path, f"{count}.txt"), 'w') as f:
            f.write(content)
        
        count += 1
