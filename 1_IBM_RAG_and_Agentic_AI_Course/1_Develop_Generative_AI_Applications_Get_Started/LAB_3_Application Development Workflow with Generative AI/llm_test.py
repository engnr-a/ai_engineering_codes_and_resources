from model import llama3_response, granite_response, mixtral_response

def call_all_models(system_prompt, user_prompt):
    llama_result = llama3_response(system_prompt, user_prompt)
    granite_result = granite_response(system_prompt, user_prompt)
    mixtral_result = mixtral_response(system_prompt, user_prompt)

    print("Llama3 Response:\n", llama_result)
    print("\nGranite Response:\n", granite_result)
    print("\nMixtral Response:\n", mixtral_result)


inquiry_prompt ="""Hi there! I just stumbled upon your new smart speaker, and I couldn’t be more excited. 
    I’ve been wanting to upgrade my home setup, and this looks like the perfect fit. I love the design, 
    and the reviews say the sound quality is outstanding. Could you please let me know if it’s compatible 
    with Spotify voice commands and whether it supports multi-room audio? I’m planning to buy two, 
    one for the living room and another for my office, so I’d also love to know if you have any bundle 
    discounts available."""
system_prompt = """You are a helpful assistant who provides concise and accurate answers to help sales representatives
    respond and react to inquiries. Given the following inquiry, analyze it and return a JSON object with:
    - summary: a brief summary of the user’s message
    - sentiment: sentiment conveyed in the inquiry
    - response: a concise suggested reply for the sales representative
    - category: the category of the inquiry (e.g., billing, technical, general, sales)
    - action: the recommended next step for the support representative
    """

if __name__ == "__main__":
    call_all_models(system_prompt, inquiry_prompt)