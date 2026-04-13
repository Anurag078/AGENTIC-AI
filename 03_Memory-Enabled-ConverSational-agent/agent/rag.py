from sentence_transformers import SentenceTransformer
import chromadb
import os

class RAGRetriever:
    def __init__(self, collection_name="conversations", embedding_model="all-MiniLM-L6-v2"):
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        # Initialize ChromaDB client and collection
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_document(self, doc_id, text, metadata=None):
        # Ensure metadata is a non-empty dict
        if not metadata or not isinstance(metadata, dict) or len(metadata) == 0:
            metadata = {"source": "user"}
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata or {}]
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Return top_k relevant documents
        return results['documents'][0] if results['documents'] else []

    def augment_prompt(self, query, top_k=3):
        context_docs = self.retrieve(query, top_k=top_k)
        context = "\n".join(context_docs)
        augmented_prompt = f"Context:\n{context}\n\nUser Query: {query}"
        return augmented_prompt

# Example usage
if __name__ == "__main__":
    rag = RAGRetriever()
    rag.add_document("1", "Python is a programming language.", {"source": "wiki"})
    rag.add_document("2", "Agentic AI agents can remember context.", {"source": "docs"})
    prompt = rag.augment_prompt("What is agentic AI?")
    print(prompt)