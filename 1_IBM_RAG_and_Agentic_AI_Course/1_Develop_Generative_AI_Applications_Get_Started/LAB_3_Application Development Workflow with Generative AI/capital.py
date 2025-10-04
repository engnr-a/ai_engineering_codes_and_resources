from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames



credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}
granite_model_id = 'ibm/granite-3-3-8b-instruct'
llama_model_id ='meta-llama/llama-3-3-70b-instruct'
mistral_model_id ='mistralai/mistral-medium-2505'
model = ModelInference(
    model_id=mistral_model_id,
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

llama_prompt_with_special_token = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
granite_prompt_with_special_token = """
<|system|>
You are an expert assistant who provides concise and accurate answers.
<|user|>
What is the capital of France?
<|assistant|>"""

mixtral_prompt_with_special_token = """
<s>[INST]
You are a highly specialized assistant. You only provide answers based on facts and logic. [/INST]

[INST] What is the capital of France? [/INST] </s>"""

print(model.generate(mixtral_prompt_with_special_token)['results'][0]['generated_text'])