import os 
import logging 
from dotenv import load_dotenv 


logging .basicConfig (
level =logging .INFO ,
format ="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger =logging .getLogger ("IntelliAgent")


load_dotenv ()


VECTOR_STORE_DIR =os .getenv ("VECTOR_STORE_DIR",".intelliagent_db")
DEFAULT_LLM_PROVIDER =os .getenv ("DEFAULT_LLM_PROVIDER","groq").lower ()
DEFAULT_EMBEDDING_PROVIDER =os .getenv ("DEFAULT_EMBEDDING_PROVIDER","openai").lower ()


DEFAULT_MODELS ={
"openai":os .getenv ("DEFAULT_LLM_MODEL","gpt-4o-mini"),
"groq":os .getenv ("DEFAULT_LLM_MODEL","llama-3.3-70b-versatile"),
}

def get_llm (provider :str =None ,model :str =None ,temperature :float =0.0 ,**kwargs ):
    """
    Factory function to initialize a LangChain Chat Model based on the provider and model.
    """
    provider =(provider or DEFAULT_LLM_PROVIDER ).lower ()
    model =model or DEFAULT_MODELS .get (provider )

    logger .info (f"Initializing LLM provider: '{provider }', model: '{model }'")

    if provider =="openai":
        api_key =os .getenv ("OPENAI_API_KEY")
        if not api_key :
            raise ValueError ("OPENAI_API_KEY is not set in environment or .env file.")
        from langchain_openai import ChatOpenAI 
        return ChatOpenAI (model =model ,temperature =temperature ,openai_api_key =api_key ,**kwargs )

    elif provider =="anthropic":
        api_key =os .getenv ("ANTHROPIC_API_KEY")
        if not api_key :
            raise ValueError ("ANTHROPIC_API_KEY is not set in environment or .env file.")
        from langchain_anthropic import ChatAnthropic 
        return ChatAnthropic (model =model ,temperature =temperature ,anthropic_api_key =api_key ,**kwargs )

    elif provider =="groq":
        api_key =os .getenv ("GROQ_API_KEY")
        if not api_key :
            raise ValueError ("GROQ_API_KEY is not set in environment or .env file.")
        from langchain_groq import ChatGroq 
        return ChatGroq (model =model ,temperature =temperature ,groq_api_key =api_key ,**kwargs )

    elif provider =="ollama":
        base_url =os .getenv ("OLLAMA_BASE_URL","http://localhost:11434")
        from langchain_community .chat_models import ChatOllama 
        return ChatOllama (model =model ,temperature =temperature ,base_url =base_url ,**kwargs )

    else :
        raise ValueError (
        f"Unsupported LLM provider '{provider }'. "
        f"Please choose from: openai, anthropic, groq, ollama."
        )
