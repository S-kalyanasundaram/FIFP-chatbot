from pymongo import MongoClient
import pandas as pd
from langchain_community.llms import Ollama
from langchain.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Choose your collection here (manually or dynamically)
COLLECTION_NAME = "mfdetails"  # 🔁 Replace

# Load MongoDB Data
client = MongoClient(MONGO_URI)
db = client["FIRE"]         # 🔁 Replace
collection = db[COLLECTION_NAME]
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

# Load the Ollama model
llm = Ollama(model="llama3")

# Create the agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

def get_agent():
    return agent
