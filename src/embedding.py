import torch
from transformers import GPT2Model, GPT2Tokenizer

class GPT2Embeddings:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def embed_documents(self, documents):
        """Generate embeddings for a list of documents."""
        return self.embed_text(documents)

    def embed_query(self, query):
        """Generate an embedding for a single query."""
        return self.embed_text([query])[0]  # Assume query is a single string

    def embed_text(self, texts):
        """General method to generate embeddings for given text, used by both document and query embedding methods."""
        with torch.no_grad():
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model(**inputs)
                # Get the mean of the last hidden state as the embedding
                mean_last_hidden_state = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(mean_last_hidden_state.cpu().numpy())
            return embeddings