from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

# Load vector store for retrieval
vectorstore = FAISS.load_local("support_docs", OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Build RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Add to LangServe - streaming works automatically
add_routes(app, rag_chain, path="/rag_agent")
