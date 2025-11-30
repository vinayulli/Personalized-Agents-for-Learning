from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langchain_core.documents import Document

def create_retriever_tool(retriever, name, description):
    def retrieve_and_format(query):
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    return Tool(
        name=name,
        description=description,
        func=retrieve_and_format
    )

from vector_store import get_vector_store

def get_tools():
    # Web Search Tool
    search_tool = TavilySearchResults(max_results=3)
    
    # Knowledge Base Tool (Qdrant)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base_retriever",
        "Search for information in the knowledge base about the specific topic."
    )
    
    return [search_tool, retriever_tool]

