import os 
import sys 


sys .path .append (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))

from agents .ingestion import IngestionAgent 
from graph .workflow import run_pipeline 
from vector_store .faiss_store import clear_vector_store 
from config .settings import logger 

def main ():
    logger .info ("=== Starting IntelliAgent Verification Test ===")


    mock_dir ="mock_docs"
    os .makedirs (mock_dir ,exist_ok =True )
    mock_file =os .path .join (mock_dir ,"ai_contributions.txt")

    mock_content ="""
    Document Title: Modern AI Innovations and Breakthroughs
    Author: Dr. Sarah Jenkins
    Date: October 2025
    
    The main contribution of the paper is the creation of a brand new architecture called the "Sparse Transformer Gate" (STG).
    This architecture reduces computational latency by 45% while retaining 98.7% accuracy across standard benchmarks.
    Additionally, the research highlights that reinforcement learning from human feedback (RLHF) was applied using a novel 
    three-stage alignment process, which significantly decreases conversational toxicities compared to traditional RLHF approaches.
    
    Furthermore, in FY2024, the primary AI lab responsible for this research reported an impressive revenue growth of 124% 
    due to cloud deployment licenses and enterprise subscriptions, reaching a total segment revenue of $8.2 billion.
    """

    with open (mock_file ,"w",encoding ="utf-8")as f :
        f .write (mock_content .strip ())

    logger .info (f"Created mock document: {mock_file }")


    logger .info ("Triggering Ingestion Agent...")
    try :

        clear_vector_store ()

        ingester =IngestionAgent (chunk_size =400 ,chunk_overlap =50 )
        ingester .ingest ([mock_file ])
        logger .info ("Ingestion completed successfully.")
    except Exception as e :
        logger .error (f"Ingestion failed: {e }")
        sys .exit (1 )


    query ="What is the main contribution of the paper and what was the revenue growth in FY2024?"
    logger .info (f"Querying: '{query }'")

    try :
        result =run_pipeline (query ,history =[])

        logger .info ("\n=== IntelliAgent Pipeline Output ===")
        logger .info (f"Query: {result ['query']}")
        logger .info (f"Response:\n{result ['response']}")
        logger .info (f"Grounded Status (from Evaluator): {result ['eval_grounded']}")
        logger .info (f"Retry Count: {result ['retry_count']}")

        logger .info ("\n=== Sources / Citations ===")
        for idx ,src in enumerate (result ['sources']):
            logger .info (f"Source [{idx +1 }]: {src ['filename']} | Page: {src ['page']}")
            logger .info (f"Snippet preview: {src ['snippet']}")

        logger .info ("\n==================================")


        assert result ['eval_grounded']is True ,"Pipeline failed grounding evaluation check!"
        assert len (result ['sources'])>0 ,"No sources retrieved!"
        logger .info ("✅ Verification test passed successfully! StateGraph operates perfectly.")

    except Exception as e :
        logger .error (f"Pipeline execution or verification failed: {e }")
        sys .exit (1 )
    finally :

        if os .path .exists (mock_file ):
            os .remove (mock_file )
        if os .path .exists (mock_dir ):
            os .rmdir (mock_dir )

        clear_vector_store ()
        logger .info ("Cleaned up mock files and indices.")

if __name__ =="__main__":
    main ()
