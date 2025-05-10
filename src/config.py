# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API settings
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# LINE Notify settings
LINE_CHANNEL_TOKEN = os.getenv("LINE_CHANNEL_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")
ENABLE_LINE_NOTIFY = os.getenv("ENABLE_LINE_NOTIFY", "True").lower() == "true"

# Trading parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LEVERAGE = 10
INITIAL_BALANCE = 15000  # JPY

# Strategy parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 3.0
POSITION_SIZE_PERCENT = 0.04

# Phase management
PHASE_THRESHOLDS = [25000, 40000, 50000]  # JPY
PHASE_LOT_FACTOR = {1: 1.0, 2: 1.0, 3: 1.0}

# File paths
DATA_DIR = "data"
BALANCE_FILE = "balance.json"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Analysis target cryptocurrency list
COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "DOT", "AVAX", "LINK", "MATIC",
    "DOGE", "SHIB", "UNI", "AAVE", "GRT",
    "RNDR", "INJ", "ARB", "OP", "IMX"
] 