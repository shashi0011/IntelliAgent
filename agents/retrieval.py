from typing import Dict ,Any ,List 
from vector_store .hybrid_retriever import load_hybrid_retriever 
from config .settings import logger 

class RetrievalAgent :
    """
    Retrieval Agent responsible for pulling user query, running hybrid search,
    and returning top chunks to be stored in the graph state.
    """
    def __init__ (self ,top_k :int =5 ):
        self .top_k =top_k 

    def retrieve (self ,state :Dict [str ,Any ])->Dict [str ,Any ]:
        """
        Retrieval node execution function.
        """
        query =state .get ("query","")
        logger .info (f"Retrieval Agent: Searching for query: '{query }'")


        retriever =load_hybrid_retriever (top_k =self .top_k )
        if retriever is None :
            logger .warning ("Retrieval Agent: No hybrid retriever found on disk. Returning empty context.")
            return {"retrieved_chunks":[],"sources":[]}

        try :
            docs =retriever .invoke (query )
        except Exception as e :
            logger .error (f"Error executing hybrid retrieval: {e }")
            docs =[]


        serialized_chunks =[]
        sources =[]
        seen_sources =set ()

        for doc in docs :
            chunk_data ={
            "content":doc .page_content ,
            "metadata":doc .metadata 
            }
            serialized_chunks .append (chunk_data )


            source_name =doc .metadata .get ("source","Unknown")
            page_num =doc .metadata .get ("page",1 )
            source_key =f"{source_name }_page_{page_num }"

            if source_key not in seen_sources :
                seen_sources .add (source_key )
                sources .append ({
                "filename":source_name ,
                "page":page_num ,
                "snippet":doc .page_content [:200 ]+"..."
                })

        logger .info (f"Retrieval Agent: Retrieved {len (serialized_chunks )} grounded chunks.")
        return {
        "retrieved_chunks":serialized_chunks ,
        "sources":sources 
        }
