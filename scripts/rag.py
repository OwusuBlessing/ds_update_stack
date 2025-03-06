import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import openai
from openai import OpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from a .env file 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Define a request model
class QuestionRequest(BaseModel):
    question: str

@app.post("/update-stack/v1/ask")
def ask_question(request: QuestionRequest):
    try:
        # Use the RAG system to get the answer
        calculation_data = get_answer(rag_chain, request.question)
        final_response = response_agent(request.question, calculation_data)
        return {"answer": final_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def setup_rag_system(file_path, llm, embedding_function):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load_and_split()
    
    index = faiss.IndexFlatL2(len(embedding_function.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=docs)
    
    retriever = vector_store.as_retriever()
    
    system_prompt = (
        "You are an AI assistant that provides investment insights. "
        "Use the Relevant Data to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def get_answer(rag_chain, user_input):
    answer = rag_chain.invoke({"input": user_input})
    return answer['answer']

def response_agent(user_question, calculation_data):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                     You are in a multi-agent system where your primary role is to provide investment insights.  
                     Based on calculation data from another agent,
                     your primary goal is to carefully look at the user question and the calculation data from the user 
                     and provide the response in a more direct and concise manner without mentioning calculation jargons.
                     
                     Example:
                     If the user asks, 'Which company is better in terms of ROI, Company A or Company B?',
                     the first agent might respond with calculations and comparisons. However, your role is to provide a
                     clear and direct answer like: 'Company B is better than Company A in terms of ROI.'
                """
            },
            {"role": "user", "content": f"""User question: {user_question}\nCalculation data: {calculation_data}"""}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Initialize OpenAI LLM and Embeddings
    llm = ChatOpenAI(model="gpt-4o-mini")  # Use the appropriate model
    embedding_function = OpenAIEmbeddings()

    # Set up the RAG system
    file_path = r'/Users/macbook/Downloads/UpdateAI_training-company-data.md'
    rag_chain = setup_rag_system(file_path, llm, embedding_function)

   
