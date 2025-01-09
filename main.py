import streamlit as st
from pandas.core.indexers import utils
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get("GROQ_API_KEY")
)

def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)
lr_model = load_model('lr_model.pkl')

def prepare_input(VIX, BDIY, DXY, USGG30YR, GT10, GTDEM10Y):
  
  input_dict = {
    'VIX': VIX,
    'BDIY': BDIY,
    'DXY': DXY,
    'USGG30YR': USGG30YR,
    'GT10': GT10,
    'GTDEM10Y': GTDEM10Y
  }
  
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):

  probabilities = {
    'Logistic Regression': lr_model.predict_proba(input_df)[0][1]
  }
  avg_probability = np.mean(list(probabilities.values()))
  
  st.markdown("### Logistic Regression Model Probability")
  for model, prob in probabilities.items():
    st.write(f"{model} {prob}")
  st.write(f"Probability: {avg_probability}")

  return avg_probability


def explain_prediction(probability, input_dict, Date):

  prompt = f"""You are an expert Financial Analyst and you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that on the date, {Date}, the market has a {round(probability * 100, 1)}% probability of being an anomaly, based on the information provided below.

  Here is the selected date's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting a market anomaly date:

      Feature |	Importance
  -----------------------------
  GTDEM10Y	|  1.889219
  VIX  |  1.694029
  GT10  |  0.367271
  DXY  |  -0.137192
  BDIY  |  -0.454736
  USGG30YR  |  -1.906397

  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for anomaly dates:
  {df[df['Y'] == 1].describe()}

  Here are summary statistics for non-anomaly dates:
  {df[df['Y'] == 0].describe()}

  - If the date has over a 30% risk of being a market anomaly, generate a 3 sentence explanation of why it may be an anomaly.
  - If the date has less than a 30% risk of being a market anomaly, generate a 3 sentence explanation of why it might not be a sign of a market anomaly.
  - Your explanation should be based on the selected date's information, the summary statistics of anomaly and non-anomaly dates, and the feature importances provided.

  Do not use first person wording such as words like "I".

  """

  print("EXPLANATION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-11b-vision-preview",
    messages=[{
      "role": "user",
      "content": prompt
    }],
  )
  return raw_response.choices[0].message.content

st.title("Atul's Market Anomaly Predictor")

df = pd.read_csv("FinancialMarketData.csv")

dates = [f"{row['Data']}" for _, row in df.iterrows()]

selected_date_option = st.selectbox("Select a date", dates)

if selected_date_option:

  selected_date = selected_date_option
  
  print("Date", selected_date)

  selected_date = df.loc[df['Data'] == selected_date].iloc[0]


  col1, col2 = st.columns(2)

  with col1:

    VIX = st.number_input(
      "VIX",
      min_value = 5.0,
      max_value= 80.0,
      value = (selected_date['VIX']))

    BDIY = st.number_input(
      "BDIY",
      min_value = 200.0,
      max_value = 12000.0,
      value = (selected_date['BDIY']))

    DXY = st.number_input(
      "DXY",
      min_value = 60.0,
      max_value=125.0,
      value = (selected_date['DXY']))

  with col2:

    USGG30YR = st.number_input(
      "USGG30YR",
      min_value = 0.0,
      max_value = 8.0,
      value = (selected_date['USGG30YR']))

    GT10 = st.number_input(
      "GT10",
      min_value = 0.0,
      max_value = 8.0,
      value = (selected_date['GT10']))
    
    GTDEM10Y = st.number_input(
      "GTDEM10Y",
      min_value = -1.0,
      max_value = 7.0,
      value = (selected_date['GTDEM10Y']))
    
  input_df, input_dict = prepare_input(VIX, BDIY, DXY, USGG30YR, GT10, GTDEM10Y)

  avg_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(avg_probability, input_dict, selected_date['Data'])

  st.markdown("---")

  st.subheader("Explanation of Prediction")

  st.markdown(explanation)
