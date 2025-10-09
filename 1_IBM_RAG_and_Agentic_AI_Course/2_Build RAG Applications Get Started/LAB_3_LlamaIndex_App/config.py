"""Configuration settings for the Icebreaker Bot."""

import os
import dotenv

############################################
# Load the environment variables from the .env file
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

model_id = "meta-llama/llama-3-3-70b-instruct"


########################################################################
# IBM watsonx.ai settings
# The URL for the Frankfurt region
WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"

# The project_id 
WATSONX_PROJECT_ID = "1d0fc49e-843a-4257-8f38-e07e1268b0a7"

# Model settings
LLM_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr"

# ProxyCurl API settings
PROXYCURL_API_KEY = ""  # Replace with your API key

# Mock data URL
MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

# Query settings
SIMILARITY_TOP_K = 5 # Retrieve more chunks for more comprehensive answers
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings
CHUNK_SIZE = 500 # Smaller chunks for more granular retrieval

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""