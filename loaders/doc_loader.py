import os 
from typing import List ,Dict ,Any 
from langchain_core .documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from config .settings import logger 

def load_pdf_pymupdf (file_path :str )->List [Document ]:
    """
    Extract text page-by-page from a PDF file using PyMuPDF (fitz).
    """
    import fitz 
    docs =[]
    filename =os .path .basename (file_path )

    try :
        with fitz .open (file_path )as pdf :
            for page_num in range (len (pdf )):
                page =pdf [page_num ]
                text =page .get_text ()


                metadata ={
                "source":filename ,
                "page":page_num +1 ,
                "file_path":file_path 
                }
                docs .append (Document (page_content =text ,metadata =metadata ))
        logger .info (f"Successfully loaded {len (docs )} pages from {filename } using PyMuPDF.")
    except Exception as e :
        logger .error (f"Error loading {filename } with PyMuPDF: {e }")
        raise e 

    return docs 

def load_pdf_pdfplumber (file_path :str )->List [Document ]:
    """
    Extract text page-by-page from a PDF file using pdfplumber as a fallback.
    """
    import pdfplumber 
    docs =[]
    filename =os .path .basename (file_path )

    try :
        with pdfplumber .open (file_path )as pdf :
            for page_num ,page in enumerate (pdf .pages ):
                text =page .extract_text ()or ""

                metadata ={
                "source":filename ,
                "page":page_num +1 ,
                "file_path":file_path 
                }
                docs .append (Document (page_content =text ,metadata =metadata ))
        logger .info (f"Successfully loaded {len (docs )} pages from {filename } using pdfplumber.")
    except Exception as e :
        logger .error (f"Error loading {filename } with pdfplumber: {e }")
        raise e 

    return docs 

def load_pdf (file_path :str )->List [Document ]:
    """
    General PDF loader that tries PyMuPDF first, then pdfplumber as fallback.
    """
    try :
        return load_pdf_pymupdf (file_path )
    except ImportError :
        logger .warning ("PyMuPDF (fitz) is not installed. Falling back to pdfplumber.")
        return load_pdf_pdfplumber (file_path )
    except Exception as e :
        logger .warning (f"PyMuPDF failed, trying pdfplumber. Error: {e }")
        try :
            return load_pdf_pdfplumber (file_path )
        except Exception as e2 :
            logger .error (f"Both PDF loaders failed for {file_path }. Error: {e2 }")
            raise RuntimeError (f"Could not load PDF file: {e2 }")

def load_txt (file_path :str )->List [Document ]:
    """
    Load text from a TXT file.
    """
    filename =os .path .basename (file_path )
    try :
        with open (file_path ,"r",encoding ="utf-8",errors ="ignore")as f :
            content =f .read ()

        metadata ={
        "source":filename ,
        "page":1 ,
        "file_path":file_path 
        }
        logger .info (f"Successfully loaded text from TXT file {filename }.")
        return [Document (page_content =content ,metadata =metadata )]
    except Exception as e :
        logger .error (f"Error loading TXT file {filename }: {e }")
        raise RuntimeError (f"Could not load TXT file: {e }")

def load_document (file_path :str )->List [Document ]:
    """
    Detects file type and loads the document. Supports PDF and TXT.
    """
    ext =os .path .splitext (file_path )[1 ].lower ()
    if ext ==".pdf":
        return load_pdf (file_path )
    elif ext in [".txt",".md"]:
        return load_txt (file_path )
    else :
        raise ValueError (f"Unsupported file format: {ext }. Only PDF and TXT are supported.")

def split_documents (documents :List [Document ],chunk_size :int =800 ,chunk_overlap :int =150 )->List [Document ]:
    """
    Splits LangChain Document objects into smaller chunks.
    Assures chunk size and overlap align with production guidelines.
    """
    splitter =RecursiveCharacterTextSplitter (
    chunk_size =chunk_size ,
    chunk_overlap =chunk_overlap ,
    length_function =len ,
    separators =["\n\n","\n"," ",""]
    )

    chunks =splitter .split_documents (documents )


    for idx ,chunk in enumerate (chunks ):
        source =chunk .metadata .get ("source","unknown")
        page =chunk .metadata .get ("page",1 )
        chunk .metadata ["chunk_id"]=f"{source }_p{page }_c{idx }"

    logger .info (f"Split {len (documents )} document pages into {len (chunks )} text chunks.")
    return chunks 
