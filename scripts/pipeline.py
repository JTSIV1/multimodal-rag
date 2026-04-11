

import torch
from vlm import VLM
from rag import RAGRetriever


class MMRAGPipeline:
    def __init__(self,
                 vlm_model_id, 
                 vlm_checkpoint_path,
                 rag = False,
                 retriever_model_id = None,
                 retriever_persist_dir = None,
                 orc = False,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 ):
        

        self.vlm = VLM(vlm_model_id, vlm_checkpoint_path)
        
        if rag:
            self.rag = RAGRetriever(
                model_id=retriever_model_id, 
                persist_dir=retriever_persist_dir, 
                device=device
            )


    def query_to_message(self, query, image_path=None):
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": query})
        return {"role": "user", "content": content}
        
    