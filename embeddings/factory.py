import os 
from langchain_core .embeddings import Embeddings 
from config .settings import DEFAULT_EMBEDDING_PROVIDER ,logger 

def get_embedding_model (provider :str =None ,**kwargs )->Embeddings :
    """
    Factory function to initialize and return an embedding model.
    Falls back gracefully to local HuggingFace embeddings if API keys are missing.
    """
    provider =(provider or DEFAULT_EMBEDDING_PROVIDER ).lower ()

    logger .info (f"Initializing embedding model provider: '{provider }'")

    if provider =="openai":
        api_key =os .getenv ("OPENAI_API_KEY")
        if api_key :
            from langchain_openai import OpenAIEmbeddings 

            model =kwargs .get ("model","text-embedding-3-small")
            return OpenAIEmbeddings (openai_api_key =api_key ,model =model ,**kwargs )
        else :
            logger .warning (
            "OPENAI_API_KEY not found in environment. "
            "Falling back to local HuggingFace embeddings (all-MiniLM-L6-v2)..."
            )
            provider ="huggingface"

    if provider =="huggingface":
        try :
            from langchain_community .embeddings import HuggingFaceEmbeddings 
            model_name =kwargs .get ("model_name","sentence-transformers/all-MiniLM-L6-v2")
            logger .info (f"Loading local HuggingFaceEmbeddings model: {model_name }")
            return HuggingFaceEmbeddings (model_name =model_name ,**kwargs )
        except Exception as e :
            logger .error (f"Failed to load HuggingFaceEmbeddings: {e }")
            raise e 

    elif provider =="ollama":
        try :
            from langchain_community .embeddings import OllamaEmbeddings 
            model_name =kwargs .get ("model_name","llama3")
            base_url =os .getenv ("OLLAMA_BASE_URL","http://localhost:11434")
            logger .info (f"Loading OllamaEmbeddings model: {model_name } from {base_url }")
            return OllamaEmbeddings (model =model_name ,base_url =base_url ,**kwargs )
        except Exception as e :
            logger .error (f"Failed to load OllamaEmbeddings: {e }")
            raise e 

    else :
        raise ValueError (
        f"Unsupported embedding provider '{provider }'. "
        f"Please choose from: openai, huggingface, ollama."
        )
