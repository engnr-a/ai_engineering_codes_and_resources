from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor


from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
#from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_ibm import WatsonxLLM

import os
import dotenv

# Load the environment variables from the .env file
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

model_id = "meta-llama/llama-3-3-70b-instruct"

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.8, # this randomness or creativity of the model's responses
}

# The URL for the Frankfurt region
url = "https://eu-de.ml.cloud.ibm.com"

# The project_id 
project_id = "1d0fc49e-843a-4257-8f38-e07e1268b0a7"

llama_llm = WatsonxLLM(
        model_id=model_id,
        url=url,                
        apikey=API_KEY, 
        project_id=project_id,
        params=parameters
    )

python_repl = PythonREPL()
python_repl_tool = Tool(
    name="Python Executor",
    func=python_repl.run,
    description="A powerful tool that can execute any Python code. Use it to perform calculations, manipulate files, or any other Python task. Input should be valid Python code."
)

@tool
def search_weather(location: str):
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API
    return f"The weather in {location} is currently sunny and 720000°F."

# Create a toolkit (collection of tools)
tools = [python_repl_tool, search_weather]

# Create the ReAct agent prompt template
# The ReAct prompt needs to instruct the model to follow the thought-action-observation pattern
prompt_template = """You are an agent who has access to the following tools:

{tools}

The available tools are: {tool_names}

To use a tool, please use the following format:
```
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
```

After you use a tool, the observation will be provided to you:
```
Observation: result of the tool
```

Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
When you have the final answer, respond in this format:
```
Thought: I know the answer
Final Answer: the final answer to the original query
```

Remember, when using the Python Calculator tool, the input must be valid Python code.

Begin!

Question: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(prompt_template)


# Create the agent
agent = create_react_agent(
    llm=llama_llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

goal = "Create a text file named 'agent_demo.txt' and write the sentence: 'This file was created by a LangChain agent deciding on its own.' Start the file with current date and time in human friendly form."
result = agent_executor.invoke({"input":goal})

print(result['output'])
