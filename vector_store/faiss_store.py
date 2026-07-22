import os 
import shutil 
from typing import List 
from langchain_community .vectorstores import FAISS 
from langchain_core .documents import Document 
from embeddings .factory import get_embedding_model 
from config .settings import VECTOR_STORE_DIR ,logger 

def get_vector_store_path ()->str :
    """
    Returns the path where the FAISS vector store is saved.
    """
    return VECTOR_STORE_DIR 

def save_vector_store (db :FAISS ,path :str =None )->None :
    """
    Saves the FAISS index to the local filesystem.
    """
    save_path =path or get_vector_store_path ()
    try :
        db .save_local (save_path )
        logger .info (f"Successfully persisted FAISS index to {save_path }")
    except Exception as e :
        logger .error (f"Error saving FAISS vector store to {save_path }: {e }")
        raise e 

def load_vector_store (path :str =None )->FAISS :
    """
    Loads the FAISS index from the local filesystem.
    """
    load_path =path or get_vector_store_path ()
    faiss_path =os .path .join (load_path ,"index.faiss")
    pkl_path =os .path .join (load_path ,"index.pkl")
    if not os .path .exists (faiss_path )or not os .path .exists (pkl_path ):
        logger .warning (f"No FAISS index found at {load_path }")
        return None 

    try :
        embeddings =get_embedding_model ()
        db =FAISS .load_local (
        load_path ,
        embeddings ,
        allow_dangerous_deserialization =True 
        )
        logger .info (f"Successfully loaded FAISS index from {load_path }")
        return db 
    except Exception as e :
        logger .error (f"Error loading FAISS vector store from {load_path }: {e }")
        return None 

def build_vector_store (chunks :List [Document ],path :str =None )->FAISS :
    """
    Builds a FAISS index from a list of document chunks and saves it locally.
    """
    save_path =path or get_vector_store_path ()
    logger .info (f"Building FAISS vector store with {len (chunks )} chunks...")

    try :
        embeddings =get_embedding_model ()
        db =FAISS .from_documents (chunks ,embeddings )
        save_vector_store (db ,save_path )
        return db 
    except Exception as e :
        logger .error (f"Error building FAISS vector store: {e }")
        raise e 

def clear_vector_store (path :str =None )->None :
    """
    Clears the local vector store directory.
    """
    clear_path =path or get_vector_store_path ()
    if os .path .exists (clear_path ):
        try :
            shutil .rmtree (clear_path )
            logger .info (f"Successfully cleared vector store directory: {clear_path }")
        except Exception as e :
            logger .error (f"Error clearing vector store directory: {e }")
            raise e 
