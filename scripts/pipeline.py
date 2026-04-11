

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
            self.use_rag = True
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
        
    
    def generate(self, query, image_path=None, sampling_params={"max_new_tokens": 1024}):

        if self.use_rag:
            retrieved_docs = self.rag.search(query, num_patches=1000, top_k=5)
            print(f"Retrieved {len(retrieved_docs)} documents from RAG.")
            # For simplicity, we just take the top retrieved document's image for VLM input
            if retrieved_docs:
                image_path = retrieved_docs[0][2]  

        message = self.query_to_message(query, image_path)
        response = self.vlm.generate([message], sampling_params)
        return response["text"]
    