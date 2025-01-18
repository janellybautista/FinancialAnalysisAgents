import numpy as np
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt

from openai import OpenAI
# from config import openai_api_key

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


client = OpenAI(api_key = openai_api_key)
GPT_MODEL = 'gpt-4o-mini'


# store user query and agent response
query_response_dict = {}


@app.get("/stock_data/")
def stock_data(user_query):

    """
    Get stock ticker from User's query 
    
    Arg:
        user_query (str): The query asking about a company's stock analysis, e.g., "Please analyze NVIDIA".
    Return:
        str: The corresponding stock ticker symbol, e.g., "NVDA".
    """

    user_prompt = (
        "<instructions>You are a professional financial analyst."
        "Your role is to respond to the user's queries by providing the correct stock ticker based on the company name."
        "Please provide the stock ticker symbol based on the company the user mentions, such as 'NVDA' for NVIDIA or 'GOOGL' for Google."
        "Only return the ticker symbol as the answer, with no additional information. "
        "For example, if the user asks about 'NVIDIA', return 'NVDA'; if the user asks about 'Google', return 'GOOGL'."
        f"Please be concise and direct in your response. The user's query is: {user_query}>\n"
    )

    ticker_response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": user_prompt}]
    )

    stock_ticker = ticker_response.choices[0].message.content


    # defult 1 year period
    start_date = "2024-01-17"
    end_date = "2025-01-17"

    # calculate metrics using yfinance
    stock = yf.Ticker(stock_ticker)
    stock_history = stock.history(start = start_date, end = end_date, interval = "1d")

    stock_history['RSI_5'] = ta.rsi(stock_history['Close'], length = 5)
    bollinger = ta.bbands(stock_history['Close'], length = 5, std = 2)
    stock_history = stock_history.join(bollinger)

    closing_prices = stock_history['Close']
    max_price = closing_prices.max()
    min_price = closing_prices.min()
    latest_price = closing_prices[-1]
    daily_returns = closing_prices.pct_change()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    annualized_vol = daily_returns.std() * np.sqrt(252)
    relative_strength_index = stock_history['RSI_5']
    bollinger_upper = stock_history['BBU_5_2.0']
    bollinger_lower = stock_history['BBL_5_2.0']

    structured_prompt = (
            "<instructions>You are a professional financial analyst specializing in time series and technical analysis."
            "Your role is to analyze the provided stock data and suggest a recommended buy price and target sell price. "
            "Take into account key technical indicators, historical trends, and risk considerations when making your recommendations. "
            "Please provide professional and clear insights while maintaining a friendly and respectful attitude.>\n"

            "<data>\n"
            f"Stock Analysis for {stock_ticker}:\n"
            f"- Date Range: {start_date} to {end_date}\n"
            f"- Closing Prices: {closing_prices.tolist()}\n"
            f"- Maximum Closing Price: {max_price}\n"
            f"- Minimum Closing Price: {min_price}\n"
            f"- Latest Closing Price: {latest_price}\n"
            f"- Sharpe Ration: {sharpe_ratio}\n"
            f"- Annualized Volatility: {annualized_vol}\n"
            f"- Relative Strength Index: {relative_strength_index.tolist()}\n"
            f"- Bollinger Upper Band: {bollinger_upper.tolist()}\n"
            f"- Bollinger Lower Band: {bollinger_lower.tolist()}\n"
            "</data>"
            )

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": structured_prompt}]
    )
    
    # store user query and agent response in dict
    query_response_dict[user_query] = response.choices[0].message.content.strip()

    return {"user_query": user_query, "response": response.choices[0].message.content}



@app.get("/fundamental_analysis/")
def fundamental_analysis(user_query):

    user_prompt = (
        "<instructions>You are a professional financial analyst."
        "Your role is to respond to the user's queries by providing the correct stock ticker based on the company name."
        "Please provide the stock ticker symbol based on the company the user mentions, such as 'NVDA' for NVIDIA or 'GOOGL' for Google."
        "Only return the ticker symbol as the answer, with no additional information. "
        "For example, if the user asks about 'NVIDIA', return 'NVDA'; if the user asks about 'Google', return 'GOOGL'."
        f"Please be concise and direct in your response. The user's query is: {user_query}>\n"
    )

    ticker_response = client.chat.completions.create(
        model = GPT_MODEL,
        messages = [{"role": "user", "content": user_prompt}]
    )

    stock_ticker = ticker_response.choices[0].message.content

    stock = yf.Ticker(stock_ticker)
    stock_balance_sheet = stock.balance_sheet
    stock_cash_flow = stock.cash_flow
    stock_financials = stock.financials

    fundamental_prompt = (
        "<instructions>You are a professional financial analyst specializing in company fundamental analysis. Your task is to analyze the financial health and performance of a company using its balance sheet, cash flow statement, and income statement data.\n\n"
        
        f"The user's query is: Perform a fundamental analysis of {stock_ticker} using the provided data.\n"
        
        "Based on this, please follow the steps below:\n"
        
        "1. Retrieve the following financial data:\n"
        f"- Balance Sheet ({stock_balance_sheet}): Review key metrics such as total assets, total liabilities, and shareholders' equity.\n"
        f"- Cash Flow Statement ({stock_cash_flow}): Analyze cash flows from operating, investing, and financing activities.\n"
        f"- Income Statement ({stock_financials}): Evaluate revenue, expenses, and profitability.\n\n"
        
        "2. Calculate key financial ratios to assess Company's performance and financial health:\n"
        "- Profitability ratios (e.g., gross profit margin, net profit margin, return on equity (ROE)).\n"
        "- Liquidity ratios (e.g., current ratio, quick ratio).\n"
        "- Solvency ratios (e.g., debt-to-equity ratio, interest coverage ratio).\n"
        "- Efficiency ratios (e.g., inventory turnover, asset turnover).\n\n"
        
        "3. Summarize the findings:\n"
        "- Highlight Company's strengths and weaknesses based on the data.\n"
        "- Discuss potential risks or opportunities.\n"
        "- Provide an overall assessment of whether Company appears fundamentally strong and potentially undervalued or overvalued.\n\n"
        
        "Please ensure that the analysis is clear, concise, and supported by numerical evidence derived from the data."
    )

    response = client.chat.completions.create(
    model = GPT_MODEL,
    messages=[{"role": "user", "content": fundamental_prompt}]
    )

    query_response_dict[user_query] = response.choices[0].message.content.strip()

    return {"user_query": user_query, "response": response.choices[0].message.content}