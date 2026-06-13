import requests
import os
from dotenv import load_dotenv

load_dotenv()  # ← this line loads the .env file

headers = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
}

# Add this to verify the keys are actually being read
print("Key:", os.getenv("ALPACA_API_KEY"))
print("Secret:", os.getenv("ALPACA_SECRET_KEY")[:6] + "..." if os.getenv("ALPACA_SECRET_KEY") else None)

r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
print(r.json())