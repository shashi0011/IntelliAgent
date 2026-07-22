from typing import TypedDict ,List ,Dict ,Any 

class AgentState (TypedDict ):
    """
    Typed dictionary representing the global state of the IntelliAgent pipeline.
    """
    query :str 
    history :List [Dict [str ,str ]]
    retrieved_chunks :List [Dict [str ,Any ]]
    response :str 
    eval_grounded :bool 
    eval_feedback :str 
    retry_count :int 
    sources :List [Dict [str ,Any ]]
