import os 
from typing import List 
from loaders .doc_loader import load_document ,split_documents 
from vector_store .hybrid_retriever import build_hybrid_retriever ,HybridRetriever 
from config .settings import logger 

class IngestionAgent :
    """
    Ingestion Agent responsible for taking file paths, loading documents page-by-page,
    chunking the content, creating embeddings, and storing them in the hybrid index.
    """
    def __init__ (self ,chunk_size :int =800 ,chunk_overlap :int =150 ):
        self .chunk_size =chunk_size 
        self .chunk_overlap =chunk_overlap 

    def ingest (self ,file_paths :List [str ])->HybridRetriever :
        """
        Orchestrates loading, chunking, and indexing for multiple files.
        """
        all_docs =[]

        for path in file_paths :
            if not os .path .exists (path ):
                logger .error (f"File path does not exist: {path }")
                continue 

            logger .info (f"Ingestion Agent: Extracting text from {path }...")
            try :
                pages =load_document (path )
                all_docs .extend (pages )
            except Exception as e :
                logger .error (f"Failed to ingest file {path }. Error: {e }")
                raise e 

        if not all_docs :
            raise ValueError ("No documents were successfully loaded.")


        chunks =split_documents (
        all_docs ,
        chunk_size =self .chunk_size ,
        chunk_overlap =self .chunk_overlap 
        )


        retriever =build_hybrid_retriever (chunks )

        logger .info ("Ingestion Agent: Successfully completed ingestion pipeline.")
        return retriever 
