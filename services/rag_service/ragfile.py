# services/rag_service/ragfile.py
from dotenv import load_dotenv
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details. Give only answers you are confident in. 
    Do not give information without a reference to the original document(s) or image(s). Avoid using Latex.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    History:\n{history}\n
    Answer:
    
    """
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "history"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


async def get_answer(manual, role, content, chat_history):
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory="./data",
        embedding_function=embedding,
        collection_name=manual,
    )

    docs = db.similarity_search(query=content)
    chain = get_conversational_chain()

    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    context = "\n".join([doc.page_content for doc in docs])

    response = chain.invoke(
        {
            "input_documents": docs,
            "context": context,
            "question": content,
            "history": history,
        },
        return_only_outputs=True,
    )

    return {"status": 200, "data": {"output_text": response}, "msg": "OK"}
