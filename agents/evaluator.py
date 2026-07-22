import json 
from typing import Dict ,Any 
from config .settings import get_llm ,logger 
from prompts .templates import EVALUATOR_PROMPT 

class EvaluatorAgent :
    """
    Evaluator Agent responsible for reviewing the generated response against 
    the retrieved document chunks to ensure factual grounding and source citation.
    Implements the self-correcting evaluation loop.
    """
    def __init__ (self ,provider :str =None ,model :str =None ):
        self .provider =provider 
        self .model =model 

    def evaluate (self ,state :Dict [str ,Any ])->Dict [str ,Any ]:
        """
        Evaluator node execution function.
        Audits generated response for hallucinations and citation accuracy.
        """
        query =state .get ("query","")
        response =state .get ("response","")
        retrieved_chunks =state .get ("retrieved_chunks",[])
        retry_count =state .get ("retry_count",0 )

        logger .info (f"Evaluator Agent: Evaluating response. Attempt: {retry_count +1 }")

        if not retrieved_chunks :
            logger .info ("Evaluator Agent: No retrieved chunks available. Skipping groundedness review.")
            return {
            "eval_grounded":True ,
            "eval_feedback":"No retrieved chunks available for evaluation.",
            "retry_count":retry_count 
            }


        context_str =""
        for idx ,chunk in enumerate (retrieved_chunks ):
            meta =chunk .get ("metadata",{})
            source =meta .get ("source","unknown")
            page =meta .get ("page",1 )
            context_str +=f"--- Chunk {idx +1 } | Source: {source } | Page: {page } ---\n"
            context_str +=f"{chunk .get ('content','')}\n\n"


        prompt_content =EVALUATOR_PROMPT .format (
        context =context_str ,
        query =query ,
        response =response 
        )


        try :

            llm =get_llm (
            provider =self .provider ,
            model =self .model ,
            temperature =0.0 ,
            model_kwargs ={"response_format":{"type":"json_object"}}
            )
            llm_response =llm .invoke (prompt_content )
            eval_output =llm_response .content .strip ()


            eval_data =json .loads (eval_output )

            is_grounded =eval_data .get ("grounded",False )
            citations_valid =eval_data .get ("citations_valid",False )
            feedback =eval_data .get ("feedback","No feedback provided by evaluator.")


            success =is_grounded and citations_valid 

            logger .info (f"Evaluator Agent: Grounded={is_grounded }, Citations Valid={citations_valid }. Success={success }")

            if success :
                logger .info ("Evaluator Agent: Response approved.")
                return {
                "eval_grounded":True ,
                "eval_feedback":"",
                "retry_count":retry_count 
                }
            else :
                logger .warning (f"Evaluator Agent: Response REJECTED. Reason: {feedback }")
                return {
                "eval_grounded":False ,
                "eval_feedback":feedback ,
                "retry_count":retry_count +1 
                }

        except Exception as e :
            logger .error (f"Error during Evaluator Agent execution: {e }")


            return {
            "eval_grounded":True ,
            "eval_feedback":f"Evaluator error occurred: {e }. Passing through.",
            "retry_count":retry_count 
            }
