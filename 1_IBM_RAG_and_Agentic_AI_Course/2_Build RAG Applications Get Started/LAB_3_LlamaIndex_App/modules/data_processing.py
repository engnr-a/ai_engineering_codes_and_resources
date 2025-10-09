"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional
import chromadb

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from modules.llm_interface import create_watsonx_embedding
import config

logger = logging.getLogger(__name__)

def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Splits the LinkedIn profile JSON data into nodes.
    
    Args:
        profile_data: LinkedIn profile data dictionary.
        
    Returns:
        List of document nodes.
    """
    # Convert the profile data to a JSON string
    profile_json = json.dumps(profile_data)
    
    # Create a Document object from the JSON string
    document = Document(text=profile_json)

    # Split the document into nodes using SentenceSplitter
    spliter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=20)
    nodes = spliter.get_nodes_from_documents([document], show_progress=True)
    
    logger.info(f"Created {len(nodes)} nodes from profile data")
    
    return nodes
    

def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Stores the document chunks (nodes) in a vector database.
    
    Args:
        nodes: List of document nodes to be indexed.
        
    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    
    # Get the embedding model
    embedding_model = create_watsonx_embedding()
    ##: NOTE: The commented out code below is in memory approach
    # # Create a VectorStoreIndex from the nodes
    # index = VectorStoreIndex(
    #     nodes=nodes,
    #     embed_model=embedding_model,
    #     show_progress=True
    # )
    
    # Set up the vector store and storage context
    db = chromadb.PersistentClient("chromadb_for_icebreaker")
    chroma_collection = db.get_or_create_collection('ai_engineering_tutorials')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Pass the nodes, the model, and storage context to VectorStoreIndex
    index = VectorStoreIndex(
                             nodes=nodes, 
                             embed_model=embedding_model,
                             storage_context=storage_context
                             )
    logger.info("VectorStoreIndex created successfully...")
    return index

def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.
    
    Args:
        index: VectorStoreIndex to verify.
        
    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        vector_store = index._storage_context.vector_store
        node_ids = list(index.index_struct.nodes_dict.keys())
        missing_embeddings = False
        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning(f"Node ID {node_id} has a None embedding.")
                missing_embeddings = True
            else:
                logger.debug(f"Node ID {node_id} has a valid embedding.")
        
        if missing_embeddings:
            logger.warning("Some node embeddings are missing")
            return False
        else:
            logger.info("All node embeddings are valid")
            return True
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False