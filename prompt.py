import os
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

scene_embeddings2 = np.load('scene_embeddings.npy')
file_path = os.path.join(os.getcwd(), "chunks.pkl")
df_chunks = pd.read_pickle(file_path)

# Define the dimension of the embeddings
d = scene_embeddings2.shape[1]

# Build a FAISS index using L2 distance
index = faiss.IndexFlatL2(d)
index.add(np.array(scene_embeddings2))

llm = Ollama(model="llama3.1")


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the LLMChain

qa_chain = PROMPT | llm
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_documents(query: str, k: int = 2) -> List[Dict]:
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector.astype('float32'), k)
    
    documents = []
    for idx, distance in zip(indices[0], distances[0]):
        documents.append({
            "content": df_chunks.iloc[idx]['combined_text'],
            "metadata": {"scene_id": df_chunks.iloc[idx]['Scene_ID']}
        })
    
    return documents

def format_docs(docs):
    return "\n".join(doc['content'] for doc in docs)


retriever = RunnableLambda(lambda q: get_relevant_documents(q))

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

def ask_question(question: str):
    # Generate answer
    result = rag_chain.invoke(question)
    
    # Retrieve relevant documents for display
    relevant_docs = get_relevant_documents(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result}")
    print("\nSources:")
    for doc in relevant_docs:
        print(f"- {doc['metadata']['scene_id']}: {doc['content']}")


# ask_question("What did Michael say about Jim's quarterlies?")
# ask_question("Who is Jan talking to in the meeting?")
ask_question("Who invented the “Suck It” vacuum-like invention in ‘The Office’?")