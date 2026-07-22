import os 
import pickle 
from typing import List ,Dict ,Any ,Tuple 
from langchain_core .retrievers import BaseRetriever 
from langchain_core .callbacks import CallbackManagerForRetrieverRun 
from langchain_core .documents import Document 
from langchain_community .vectorstores import FAISS 
from langchain_community .retrievers import BM25Retriever 
from vector_store .faiss_store import load_vector_store ,get_vector_store_path 
from config .settings import logger 

class HybridRetriever (BaseRetriever ):
    """
    Custom Retriever implementing Reciprocal Rank Fusion (RRF) 
    over Dense (FAISS) and Sparse (BM25) retrievers.
    """
    dense_vectorstore :FAISS 
    sparse_retriever :BM25Retriever 
    top_k :int =5 
    rrf_k :int =60 

    def _get_relevant_documents (
    self ,query :str ,*,run_manager :CallbackManagerForRetrieverRun =None 
    )->List [Document ]:
        """
        Retrieves relevant documents by combining FAISS and BM25 using RRF.
        """


        fetch_k =self .top_k *2 


        try :
            dense_docs =self .dense_vectorstore .similarity_search (query ,k =fetch_k )
        except Exception as e :
            logger .warning (f"Dense retrieval unavailable; falling back to sparse retrieval. Reason: {e }")
            dense_docs =[]


        try :

            self .sparse_retriever .k =fetch_k 
            sparse_docs =self .sparse_retriever .invoke (query )
        except Exception as e :
            logger .error (f"Error in sparse retrieval: {e }")
            sparse_docs =[]


        rrf_scores :Dict [str ,float ]={}
        doc_map :Dict [str ,Document ]={}


        def accumulate_rrf (docs :List [Document ]):
            for rank ,doc in enumerate (docs ):

                chunk_id =doc .metadata .get ("chunk_id")or doc .page_content 
                doc_map [chunk_id ]=doc 


                score =1.0 /(self .rrf_k +rank )
                rrf_scores [chunk_id ]=rrf_scores .get (chunk_id ,0.0 )+score 

        accumulate_rrf (dense_docs )
        accumulate_rrf (sparse_docs )


        sorted_chunk_ids =sorted (rrf_scores .keys (),key =lambda x :rrf_scores [x ],reverse =True )


        final_docs =[doc_map [cid ]for cid in sorted_chunk_ids [:self .top_k ]]

        logger .info (
        f"Hybrid retrieval complete: merged {len (dense_docs )} dense docs "
        f"and {len (sparse_docs )} sparse docs into top {len (final_docs )} results."
        )
        return final_docs 

def build_hybrid_retriever (chunks :List [Document ],path :str =None )->HybridRetriever :
    """
    Builds the dense FAISS index and the BM25 sparse model, 
    and saves them to the persistent directory.
    """
    save_path =path or get_vector_store_path ()
    os .makedirs (save_path ,exist_ok =True )


    from vector_store .faiss_store import build_vector_store 
    dense_db =build_vector_store (chunks ,save_path )


    logger .info ("Building BM25 sparse retriever...")
    bm25 =BM25Retriever .from_documents (chunks )

    bm25_file_path =os .path .join (save_path ,"bm25_retriever.pkl")
    try :
        with open (bm25_file_path ,"wb")as f :
            pickle .dump (bm25 ,f )
        logger .info (f"Successfully persisted BM25 retriever to {bm25_file_path }")
    except Exception as e :
        logger .error (f"Error persisting BM25 retriever: {e }")
        raise e 

    return HybridRetriever (dense_vectorstore =dense_db ,sparse_retriever =bm25 )

def load_hybrid_retriever (path :str =None ,top_k :int =5 )->HybridRetriever :
    """
    Loads FAISS and BM25 retrievers from the persistent directory 
    and instantiates the HybridRetriever.
    """
    load_path =path or get_vector_store_path ()


    dense_db =load_vector_store (load_path )
    if dense_db is None :
        return None 


    bm25_file_path =os .path .join (load_path ,"bm25_retriever.pkl")
    if not os .path .exists (bm25_file_path ):
        logger .warning (f"No BM25 retriever found at {bm25_file_path }")
        return None 

    try :
        with open (bm25_file_path ,"rb")as f :
            bm25 =pickle .load (f )
        logger .info (f"Successfully loaded BM25 retriever from {bm25_file_path }")
    except Exception as e :
        logger .error (f"Error loading BM25 retriever: {e }")
        return None 

    return HybridRetriever (dense_vectorstore =dense_db ,sparse_retriever =bm25 ,top_k =top_k )
