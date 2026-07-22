from typing import Dict ,Any ,List 
from config .settings import get_llm ,logger 
from prompts .templates import RAG_RESPONSE_PROMPT 

class ResponseAgent :
    """
    Response Agent responsible for synthesizing the retrieved context,
    chat history, and user query using the configured LLM, generating a
    highly-accurate, source-cited answer. Also handles iterative refinement.
    """
    def __init__ (self ,provider :str =None ,model :str =None ):
        self .provider =provider 
        self .model =model 

    def generate (self ,state :Dict [str ,Any ])->Dict [str ,Any ]:
        """
        Response node execution function.
        Generates or refines an answer based on current state.
        """
        query =state .get ("query","")
        history =state .get ("history",[])
        retrieved_chunks =state .get ("retrieved_chunks",[])
        prev_response =state .get ("response","")
        feedback =state .get ("eval_feedback","")
        retry_count =state .get ("retry_count",0 )

        logger .info (f"Response Agent: Generating response. Attempt: {retry_count +1 }")


        if not retrieved_chunks :
            return {
            "response":"Based on the uploaded documents, I couldn't find any relevant information to answer your question."
            }

        context_str =""
        for idx ,chunk in enumerate (retrieved_chunks ):
            meta =chunk .get ("metadata",{})
            source =meta .get ("source","unknown")
            page =meta .get ("page",1 )
            context_str +=f"--- Chunk {idx +1 } | Source: {source } | Page: {page } ---\n"
            context_str +=f"{chunk .get ('content','')}\n\n"


        history_str =""
        if history :
            for msg in history :
                role ="User"if msg .get ("role")=="user"else "IntelliAgent"
                history_str +=f"{role }: {msg .get ('content','')}\n"
        else :
            history_str ="None"


        prompt_content =RAG_RESPONSE_PROMPT .format (
        context =context_str ,
        history =history_str ,
        query =query 
        )

        if retry_count >0 and feedback :
            logger .info ("Response Agent: Integrating evaluator feedback for self-correction...")
            prompt_content +=f"\n\n=== REFINEMENT REQUEST ===\n" f"Your previous answer was rejected by the Evaluator Agent.\n" f"Previous Answer: {prev_response }\n" f"Evaluator Critique & Instructions: {feedback }\n" f"Please correct the answer while keeping it concise, direct, and grounded. " f"Use only the citations needed to verify the answer."


        try :
            llm =get_llm (provider =self .provider ,model =self .model ,temperature =0.0 )
            llm_response =llm .invoke (prompt_content )
            response_text =llm_response .content 
        except Exception as e :
            logger .error (f"Error during LLM response generation: {e }")
            response_text =f"An error occurred while generating a response from the LLM provider: {e }"

        logger .info ("Response Agent: Generation complete.")
        return {
        "response":response_text 
        }
