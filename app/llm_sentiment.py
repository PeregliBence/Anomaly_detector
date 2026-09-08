"""
LLM-powered Sentiment Engine — OpenAI, SEC EDGAR, Named Entity Recognition.

Components:
  1. OpenAI-based sentiment extraction (replaces keyword heuristics)
  2. SEC EDGAR earnings transcript fetching & analysis
  3. Named Entity Recognition for better ticker-news matching

All components gracefully degrade if API keys / dependencies are missing.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from app.config import logger, _env

# ── Conditional imports ──────────────────────────────────────────────────

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# ============================================================================
# 1. OPENAI SENTIMENT EXTRACTION
# ============================================================================

_SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analysis engine.
For each news headline/snippet, output ONLY valid JSON with these fields:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "impact": "high" | "medium" | "low",
  "category": "earnings" | "analyst" | "macro" | "regulatory" | "merger" | "product" | "other",
  "key_entities": ["TICKER1", "TICKER2"],
  "summary": "one-sentence reason"
}
Do not include any text outside the JSON object."""


def analyze_sentiment_llm(
    headlines: list[str],
    model: str = "gemini-2.0-flash",
    max_batch: int = 20,
) -> list[dict]:
    """
    Analyze financial news headlines using Google Gemini API.

    Free tier generous limits with gemini-2.0-flash.
    Falls back to empty results if API key is missing.

    Parameters
    ----------
    headlines : list of headline strings
    model : Gemini model name
    max_batch : max headlines per API call to control cost

    Returns list of dicts with sentiment analysis per headline.
    """
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set — skipping LLM sentiment")
        return []

    if not HAS_GEMINI:
        logger.warning("google-generativeai package not installed (pip install google-generativeai)")
        return []

    if not headlines:
        return []

    # Limit batch size for cost control
    headlines = headlines[:max_batch]

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    results = []

    # Process in small batches for reliability
    batch_size = 5
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {h}" for j, h in enumerate(batch))

        prompt = (
            f"{_SENTIMENT_SYSTEM_PROMPT}\n\n"
            f"Analyze these financial news headlines and return a JSON array "
            f"with one object per headline:\n\n{numbered}"
        )

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                ),
            )

            content = response.text.strip()

            # Strip markdown code fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            # Parse JSON (handle both single object and array)
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed = [parsed]
            results.extend(parsed)

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM sentiment response as JSON")
            # Try to extract JSON from response
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    results.extend(json.loads(match.group()))
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.error("Gemini sentiment analysis failed: %s", e)

    logger.info("LLM sentiment analyzed %d/%d headlines", len(results), len(headlines))
    return results


def get_aggregate_sentiment(analyses: list[dict]) -> dict:
    """
    Aggregate multiple headline sentiments into a single market view.

    Returns:
      - overall: "bullish" | "bearish" | "neutral"
      - score: -1.0 to 1.0
      - high_impact_count: number of high-impact items
      - categories: dict of category counts
    """
    if not analyses:
        return {"overall": "neutral", "score": 0.0, "high_impact_count": 0, "categories": {}}

    scores = []
    categories: dict[str, int] = {}
    high_impact = 0

    for a in analyses:
        sent = a.get("sentiment", "neutral")
        conf = float(a.get("confidence", 0.5))
        impact = a.get("impact", "low")
        cat = a.get("category", "other")

        if sent == "bullish":
            scores.append(conf)
        elif sent == "bearish":
            scores.append(-conf)
        else:
            scores.append(0.0)

        if impact == "high":
            high_impact += 1

        categories[cat] = categories.get(cat, 0) + 1

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    if avg_score > 0.15:
        overall = "bullish"
    elif avg_score < -0.15:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "score": round(avg_score, 3),
        "high_impact_count": high_impact,
        "categories": categories,
        "n_analyzed": len(analyses),
    }


# ============================================================================
# 2. SEC EDGAR — Earnings Transcript Fetching
# ============================================================================

_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index?q="
_EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_FILINGS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# User-Agent required by SEC EDGAR (they block requests without it)
_EDGAR_HEADERS = {
    "User-Agent": "AnomalyDetector/1.0 (academic research)",
    "Accept-Encoding": "gzip, deflate",
}


def _search_edgar_cik(ticker: str) -> Optional[str]:
    """Look up SEC CIK number for a ticker symbol."""
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{urllib.parse.quote(ticker)}%22&dateRange=custom&startdt=2020-01-01&forms=10-K,10-Q,8-K"
        req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                cik = hits[0].get("_source", {}).get("entity_id")
                return str(cik).zfill(10) if cik else None
    except Exception:
        pass

    # Fallback: company tickers JSON
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        logger.warning("EDGAR CIK lookup failed: %s", e)

    return None


def fetch_recent_filings(
    ticker: str,
    form_types: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    Fetch recent SEC filings metadata for a ticker.

    Returns list of dicts with: form_type, filed_date, description, url
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    cik = _search_edgar_cik(ticker)
    if not cik:
        logger.info("Could not find SEC CIK for %s", ticker)
        return []

    try:
        url = _EDGAR_FILINGS_URL.format(cik=cik)
        req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i in range(len(forms)):
            if forms[i] in form_types:
                acc_clean = accessions[i].replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_docs[i]}"
                results.append({
                    "form_type": forms[i],
                    "filed_date": dates[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": doc_url,
                })
                if len(results) >= max_results:
                    break

        logger.info("Found %d SEC filings for %s", len(results), ticker)
        return results

    except Exception as e:
        logger.warning("EDGAR filings fetch failed for %s: %s", ticker, e)
        return []


def analyze_filing_with_llm(
    filing_text: str,
    model: str = "gemini-2.0-flash",
    max_chars: int = 8000,
) -> Optional[dict]:
    """
    Analyze SEC filing text (or excerpt) with Gemini.

    Returns dict with: sentiment, key_risks, key_positives, outlook
    """
    api_key = _env("GEMINI_API_KEY")
    if not api_key or not HAS_GEMINI:
        return None

    # Truncate for token limits
    text = filing_text[:max_chars]

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)

    prompt = (
        "You are an SEC filing analyst. Analyze the following filing excerpt and "
        'return JSON: {"sentiment": "bullish"|"bearish"|"neutral", '
        '"key_risks": ["risk1", ...], "key_positives": ["pos1", ...], '
        '"outlook": "one-sentence outlook", "confidence": 0.0-1.0}\n\n'
        + text
    )

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            ),
        )
        content = response.text.strip()
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        return json.loads(content)

    except Exception as e:
        logger.warning("Gemini filing analysis failed: %s", e)
        return None


# ============================================================================
# 3. NAMED ENTITY RECOGNITION (NER) for Ticker Matching
# ============================================================================

# Regex-based NER for common financial patterns (no external dependency)

_TICKER_PATTERN = re.compile(
    r'\b([A-Z]{1,5})\b'
    r'(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|Co\.?|Group|Holdings?|Enterprises?))?',
)

_COMPANY_ALIASES: dict[str, str] = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOG", "Alphabet": "GOOG",
    "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META", "Facebook": "META",
    "Netflix": "NFLX", "Nvidia": "NVDA", "AMD": "AMD",
    "JPMorgan": "JPM", "Goldman Sachs": "GS", "Goldman": "GS",
    "Boeing": "BA", "Exxon": "XOM", "Chevron": "CVX",
    "Coinbase": "COIN", "Palantir": "PLTR",
    "S&P 500": "SPY", "S&P": "SPY", "Nasdaq": "QQQ", "Dow Jones": "DIA",
}

# Common words that look like tickers but aren't
_TICKER_BLACKLIST = {
    "THE", "FOR", "AND", "BUT", "NOT", "ARE", "WAS", "HAS", "HAD", "HIS",
    "HER", "ITS", "ALL", "ANY", "CAN", "MAY", "NEW", "OLD", "ONE", "TWO",
    "NOW", "OUT", "OUR", "OWN", "SAY", "SHE", "TOO", "USE", "HIM", "HOW",
    "MAN", "BIG", "GET", "CEO", "CFO", "IPO", "SEC", "FDA", "FBI", "ETF",
    "GDP", "CPI", "FED", "NYSE", "US", "UK", "EU", "AI", "IT", "PM", "AM",
    "VS", "A", "I", "IN", "ON", "UP", "AT", "TO", "BY", "IS", "IF", "AS",
    "SO", "NO", "GO", "DO", "BE",
}


def extract_tickers_from_text(text: str) -> list[str]:
    """
    Extract likely stock tickers from news text using regex + company alias matching.

    Returns deduplicated list of ticker symbols.
    """
    tickers = set()

    # 1. Company name matching
    text_lower = text.lower()
    for company, ticker in _COMPANY_ALIASES.items():
        if company.lower() in text_lower:
            tickers.add(ticker)

    # 2. Dollar-sign prefix ($AAPL)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
    tickers.update(dollar_tickers)

    # 3. Regex pattern for standalone uppercase words (potential tickers)
    for match in _TICKER_PATTERN.finditer(text):
        candidate = match.group(1)
        if candidate not in _TICKER_BLACKLIST and len(candidate) >= 2:
            tickers.add(candidate)

    return sorted(tickers)


def match_news_to_ticker(
    headline: str,
    body: str = "",
    target_ticker: str = "",
) -> dict:
    """
    Determine if a news article is relevant to a specific ticker.

    Returns:
      - is_relevant: bool
      - confidence: float (0-1)
      - matched_entities: list of extracted tickers
      - match_type: "direct" | "alias" | "sector" | "none"
    """
    full_text = f"{headline} {body}"
    extracted = extract_tickers_from_text(full_text)

    if target_ticker.upper() in extracted:
        return {
            "is_relevant": True,
            "confidence": 0.95,
            "matched_entities": extracted,
            "match_type": "direct",
        }

    # Check for company name alias
    for company, ticker in _COMPANY_ALIASES.items():
        if ticker == target_ticker.upper() and company.lower() in full_text.lower():
            return {
                "is_relevant": True,
                "confidence": 0.9,
                "matched_entities": extracted,
                "match_type": "alias",
            }

    # Sector match (same sector tickers mentioned)
    sector_groups = {
        "tech": {"AAPL", "MSFT", "GOOG", "META", "AMZN", "NVDA", "AMD", "NFLX", "CRM"},
        "finance": {"JPM", "GS", "BAC", "WFC", "MS", "C"},
        "energy": {"XOM", "CVX", "COP", "SLB", "OXY"},
        "crypto": {"COIN", "MARA", "RIOT"},
    }

    target_sector = None
    for sector, members in sector_groups.items():
        if target_ticker.upper() in members:
            target_sector = sector
            break

    if target_sector:
        members = sector_groups[target_sector]
        if any(t in members for t in extracted):
            return {
                "is_relevant": True,
                "confidence": 0.4,
                "matched_entities": extracted,
                "match_type": "sector",
            }

    return {
        "is_relevant": False,
        "confidence": 0.0,
        "matched_entities": extracted,
        "match_type": "none",
    }


# ============================================================================
# 4. COMBINED SENTIMENT PIPELINE
# ============================================================================

@dataclass
class SentimentReport:
    """Aggregated sentiment analysis result."""
    overall: str = "neutral"
    score: float = 0.0
    headline_analyses: list = field(default_factory=list)
    filing_analysis: Optional[dict] = None
    fear_greed: Optional[dict] = None
    high_impact_news: int = 0
    method: str = "keyword"  # "keyword" | "llm"


def run_sentiment_pipeline(
    ticker: str,
    headlines: list[str] | None = None,
    include_filings: bool = True,
) -> SentimentReport:
    """
    Run the full sentiment analysis pipeline.

    1. Analyze news headlines (LLM if available, keyword fallback)
    2. Fetch and analyze recent SEC filings (if enabled)
    3. Aggregate results

    Returns a SentimentReport with all findings.
    """
    report = SentimentReport()

    # 1. News headline sentiment
    if headlines:
        api_key = _env("GEMINI_API_KEY")
        if api_key and HAS_GEMINI:
            analyses = analyze_sentiment_llm(headlines)
            if analyses:
                report.headline_analyses = analyses
                agg = get_aggregate_sentiment(analyses)
                report.overall = agg["overall"]
                report.score = agg["score"]
                report.high_impact_news = agg["high_impact_count"]
                report.method = "llm"
            else:
                report.method = "keyword"
        else:
            report.method = "keyword"

    # 2. SEC filings
    if include_filings:
        filings = fetch_recent_filings(ticker, max_results=1)
        if filings:
            report.filing_analysis = {
                "recent_filing": filings[0],
                "form_type": filings[0].get("form_type"),
                "date": filings[0].get("filed_date"),
            }

    return report
