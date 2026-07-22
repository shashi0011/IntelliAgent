from langgraph .graph import StateGraph ,END 
from graph .state import AgentState 
from agents .retrieval import RetrievalAgent 
from agents .response import ResponseAgent 
from agents .evaluator import EvaluatorAgent 
from config .settings import logger 


retrieval_agent =RetrievalAgent (top_k =5 )
response_agent =ResponseAgent ()
evaluator_agent =EvaluatorAgent ()

def route_evaluation (state :AgentState )->str :
    """
    Conditional router that determines whether to finalize or trigger self-correction.
    """
    is_grounded =state .get ("eval_grounded",False )
    retry_count =state .get ("retry_count",0 )

    if is_grounded :
        logger .info ("LangGraph Router: Response approved. Transitioning to END.")
        return "finalize"

    if retry_count >=3 :
        logger .warning (
        f"LangGraph Router: Response rejected but reached maximum retry limit ({retry_count }). "
        "Transitioning to END to prevent infinite execution loop."
        )
        return "finalize"

    logger .info (
    f"LangGraph Router: Response rejected (Attempt {retry_count }). "
    "Routing back to Response Agent for refinement."
    )
    return "re_generate"


workflow =StateGraph (AgentState )


workflow .add_node ("retrieve",retrieval_agent .retrieve )
workflow .add_node ("generate",response_agent .generate )
workflow .add_node ("evaluate",evaluator_agent .evaluate )


workflow .set_entry_point ("retrieve")
workflow .add_edge ("retrieve","generate")
workflow .add_edge ("generate","evaluate")


workflow .add_conditional_edges (
"evaluate",
route_evaluation ,
{
"finalize":END ,
"re_generate":"generate"
}
)


compiled_graph =workflow .compile ()
logger .info ("LangGraph multi-agent RAG workflow compiled successfully.")

def run_pipeline (query :str ,history :list )->dict :
    """
    Executes the completed multi-agent RAG workflow.
    
    Parameters:
    - query: User query string
    - history: Conversational history list of dicts with role and content
    """
    initial_state ={
    "query":query ,
    "history":history ,
    "retrieved_chunks":[],
    "response":"",
    "eval_grounded":False ,
    "eval_feedback":"",
    "retry_count":0 ,
    "sources":[]
    }

    logger .info (f"LangGraph execution started for query: '{query }'")
    final_state =compiled_graph .invoke (initial_state )
    logger .info ("LangGraph execution completed.")
    return final_state 
