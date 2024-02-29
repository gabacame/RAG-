from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
#from pdf import Differential_Equations_engine

load_dotenv()

equations_path = os.path.join("data", "math-equations.csv")
equations_df = pd.read_csv(equations_path)

equations_query_engine = PandasQueryEngine(
    df=equations_df, verbose=True, instruction_str=instruction_str
)
equations_query_engine.update_prompts({"pandas_prompt" : new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=equations_query_engine, 
        metadata=ToolMetadata(
            name="equations_data",
            description="this gives information about some currently used math equations",
        ),
    ),
    '''
    QueryEngineTool(
        query_engine=Differential_Equations_engine, 
        metadata=ToolMetadata(
            name="Differential_Equations_data",
            description="this help you through Differential Equations concepts and topics from class notes, also can help you how to solve algebra problems",
        ),
    ), '''
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)