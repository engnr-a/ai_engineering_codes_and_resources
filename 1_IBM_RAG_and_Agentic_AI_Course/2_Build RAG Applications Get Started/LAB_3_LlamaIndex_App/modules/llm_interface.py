"""Module for interfacing with IBM watsonx.ai LLMs."""

import logging
from typing import Dict, Any, Optional

from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

import config
import os
import dotenv

# Load the environment variables from the .env file
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")


logger = logging.getLogger(__name__)

def create_watsonx_embedding() -> WatsonxEmbeddings:
    """Creates an IBM Watsonx Embedding model for vector representation.
    
    Returns:
        WatsonxEmbeddings model.
    """
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 500, #model with max token at 512
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id=config.EMBEDDING_MODEL_ID, #--> "ibm/slate-125m-english-rtrvr"
        url=config.WATSONX_URL,
        apikey=API_KEY, 
        project_id=config.WATSONX_PROJECT_ID,
        params=embed_params,
    )
    return watsonx_embedding

def create_watsonx_llm(
    temperature: float = 0.0,
    max_new_tokens: int = 500,
    decoding_method: str = "sample"
) -> WatsonxLLM:
    """Creates an IBM Watsonx LLM for generating responses.
    
    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).
        
    Returns:
        WatsonxLLM model.
    """
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
        GenParams.MIN_NEW_TOKENS: 130, # controls the minimum number of tokens in the generated output
        GenParams.MAX_NEW_TOKENS: 256,  # controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5 #randomness or creativity of the model's responses
    }
    

    watsonx_llm = WatsonxLLM(
        model_id=config.LLM_MODEL_ID ,
        url=config.WATSONX_URL,
        apikey=API_KEY, 
        project_id=config.WATSONX_PROJECT_ID,
        params=parameters,
    )
    return watsonx_llm


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
