"""
Ticker search functionality for the UI.
Supports local universe search and Yahoo Finance symbol lookup.
"""

TOP_TICKERS = [
    ("Apple Inc.", "AAPL"),
    ("Microsoft Corporation", "MSFT"),
    ("NVIDIA Corporation", "NVDA"),
    ("Tesla Inc.", "TSLA"),
    ("Amazon.com Inc.", "AMZN"),
]

LOCAL_UNIVERSE = [
    # US Tech
    ("Apple Inc.", "AAPL"),
    ("Microsoft Corporation", "MSFT"),
    ("NVIDIA Corporation", "NVDA"),
    ("Tesla Inc.", "TSLA"),
    ("Amazon.com Inc.", "AMZN"),
    ("Meta Platforms Inc.", "META"),
    ("Alphabet Inc. Class A", "GOOGL"),
    ("Netflix Inc.", "NFLX"),
    ("Advanced Micro Devices Inc.", "AMD"),
    ("Intel Corporation", "INTC"),
    ("Broadcom Inc.", "AVGO"),
    ("Adobe Inc.", "ADBE"),
    ("Salesforce Inc.", "CRM"),
    ("Oracle Corporation", "ORCL"),
    # US Finance
    ("JPMorgan Chase & Co.", "JPM"),
    ("Bank of America Corporation", "BAC"),
    ("Visa Inc.", "V"),
    ("Mastercard Incorporated", "MA"),
    ("Berkshire Hathaway Inc. Class B", "BRK-B"),
    ("Goldman Sachs Group Inc.", "GS"),
    # US Consumer / Industrial
    ("Coca-Cola Company", "KO"),
    ("PepsiCo Inc.", "PEP"),
    ("Exxon Mobil Corporation", "XOM"),
    ("Chevron Corporation", "CVX"),
    ("Boeing Company", "BA"),
    ("Caterpillar Inc.", "CAT"),
    # ETFs
    ("SPDR S&P 500 ETF Trust", "SPY"),
    ("Invesco QQQ Trust", "QQQ"),
    ("iShares Russell 2000 ETF", "IWM"),
    ("Vanguard Total Stock Market ETF", "VTI"),
    ("iShares MSCI Emerging Markets ETF", "EEM"),
    ("SPDR Gold Shares", "GLD"),
    ("United States Oil Fund", "USO"),
    # Indices
    ("S&P 500 Index", "^GSPC"),
    ("Dow Jones Industrial Average", "^DJI"),
    ("NASDAQ Composite", "^IXIC"),
    ("NASDAQ 100 Index", "^NDX"),
    ("DAX Performance Index", "^GDAXI"),
    ("Euro Stoxx 50", "^STOXX50E"),
    ("BUX Index (Budapest)", "^BUX"),
    ("FTSE 100 Index", "^FTSE"),
    ("Nikkei 225", "^N225"),
    # Forex
    ("EUR / USD", "EURUSD=X"),
    ("EUR / HUF", "EURHUF=X"),
    ("USD / HUF", "USDHUF=X"),
    ("GBP / USD", "GBPUSD=X"),
    ("EUR / GBP", "EURGBP=X"),
    ("USD / JPY", "USDJPY=X"),
    # Crypto
    ("Bitcoin / USD", "BTC-USD"),
    ("Ethereum / USD", "ETH-USD"),
    ("Solana / USD", "SOL-USD"),
    ("Ripple (XRP) / USD", "XRP-USD"),
    ("Litecoin / USD", "LTC-USD"),
    ("Cardano / USD", "ADA-USD"),
]


def yahoo_symbol_search(searchterm: str):
    """
    Local ticker search: returns list[tuple[label, symbol]].
    Falls back to TOP_TICKERS when no input or no match.
    """
    if not searchterm:
        return [(f"{name} ({sym})", sym) for name, sym in TOP_TICKERS]

    term = searchterm.lower().strip()
    matches = [
        (f"{name} ({sym})", sym)
        for name, sym in LOCAL_UNIVERSE
        if term in name.lower() or term in sym.lower()
    ]

    return matches if matches else [(f"{name} ({sym})", sym) for name, sym in TOP_TICKERS]
