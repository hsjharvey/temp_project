import yfinance as yf

# List of crypto tickers in AUD
cryptos = ["BTC-AUD", "ETH-AUD", "SOL-AUD", "BNB-AUD", "XRP-AUD", "^AXJO"]

# Date range
start_date = "2020-01-01"
end_date = "2025-11-25"

# Dictionary to store dataframes
crypto_data = {}

for coin in cryptos:
    df = yf.download(coin, start=start_date, end=end_date)
    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    crypto_data[coin] = df
    df["daily_return"] = df["Close"].pct_change()
    df = df.dropna()

    # Save each to CSV
    df.to_csv(f"{coin}.csv")
    print(f"Saved {coin} data with {len(df)} rows")

# Example: access BTC-AUD data
print(crypto_data["BTC-AUD"].head())