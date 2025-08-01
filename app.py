import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pandas import json_normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# ---- Page Setup ----
st.set_page_config(page_title="ML Mutual Fund Predictor", layout="centered")
st.title("📈 ML-Based Mutual Fund Predictor")
st.caption("Rank funds by model trained on performance, risk, and expense")

# ---- Load MongoDB Data ----
@st.cache_data
def load_data():
    client = MongoClient(MONGO_URI)
    db = client["FIRE"]
    collection = db["mfdetails"]
    data = list(collection.find({}, {"_id": 0}))
    return json_normalize(data)

# ---- Preprocess Data ----
def preprocess_data(df):
    if df.empty:
        return df

    for col in ["nav", "expense_ratio", "risk_rating", "fund_size"]:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["nav", "expense_ratio", "risk_rating"], inplace=True)
    return df

# ---- Train ML Model ----
@st.cache_data
def train_model(df):
    df = df.copy()

    # Feature Engineering
    features = df[["nav", "expense_ratio", "risk_rating"]]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)

    # Simulated target variable: "score" based on a weighted rule
    df["score"] = (
        0.4 * (df["nav"] / df["nav"].max()) +
        0.3 * (1 - (df["expense_ratio"] / df["expense_ratio"].max())) +
        0.3 * (1 / (df["risk_rating"] + 1))
    )

    y = df["score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df["predicted_score"] = model.predict(X)

    return df.sort_values("predicted_score", ascending=False).head(5)[
        ["scheme_name", "nav", "expense_ratio", "risk_rating", "predicted_score"]
    ], model

# ---- LangChain Agent ----
def load_agent(df):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    instructions = "You are a mutual fund advisor. Handle fund comparisons and explain results."

    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        system_message=instructions
    )

# ---- Prediction Function ----
def predict_using_model(user_input, df, model):
    if "best scheme" in user_input.lower() or "top fund" in user_input.lower():
        top_funds, _ = train_model(df)
        response = "📊 **Top ML-Predicted Mutual Funds:**\n\n"
        for i, row in top_funds.iterrows():
            response += f"- **{row['scheme_name']}**\n  NAV: ₹{row['nav']:.2f}, Expense: {row['expense_ratio']}%, Risk: {row['risk_rating']}, Score: {row['predicted_score']:.3f}\n\n"
        return response
    return None

# ---- Main App ----
try:
    df_raw = load_data()
    df = preprocess_data(df_raw)

    if df.empty:
        st.error("❌ No valid data found.")
        st.stop()

    st.success("✅ Data loaded and preprocessed.")

    # Train model once
    top_funds, model = train_model(df)

    agent = load_agent(df)

    # Initial messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! 👋 Ask things like:\n- Which scheme is best?\n- Suggest top mutual fund\n- Compare HDFC vs Axis"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a mutual fund question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = predict_using_model(user_input, df, model)
                    if not response:
                        response = agent.run(user_input)
                except Exception as e:
                    response = f"❌ Error: {e}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"❌ Error loading app: {e}")
