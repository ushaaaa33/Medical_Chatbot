import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"


from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,   # Controls randomness.
        api_key=GROQ_API_KEY
    )


# Step 2: Connect LLM with FAISS and Create chain

#  It forces the model to: Use only retrieved context, Avoid hallucination
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


qa_chain=RetrievalQA.from_chain_type(      # Creates RAG pipeline.
    llm=load_llm(),
    chain_type="stuff",   # Put all retrieved chunks into a single prompt.
    retriever=db.as_retriever(search_kwargs={'k':3}),  # Retrieve top 3 relevant chunks.
    return_source_documents=True,  # Returns source chunks used for answering.
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

 
# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})   # Runs complete RAG process
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])


