import os
from py_clob_client.client import ClobClient

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon

API_KEY = os.environ["PM_API_KEY"]
API_SECRET = os.environ["PM_API_SECRET"]
PASSPHRASE = os.environ["PM_API_PASSPHRASE"]

# まずは“確実に存在する” token_id を1つ指定
# （次で Gamma から自動取得に戻す）
TOKEN_ID = "0x3b7f4f4a2c3f0c0f9c7c0e5c1b7b1a0e0a9a9a9a"  # ダミーではなく、次で差し替える

def main():
    client = ClobClient(
        host=HOST,
        key=API_KEY,
        api_secret=API_SECRET, 
        passphrase=PASSPHRASE,
        chain_id=CHAIN_ID,
    )

    # BUY側の現在価格を取得（署名付き）
    px = client.get_price(token_id=TOKEN_ID, side="BUY")
    print("PRICE:", px)

if __name__ == "__main__":
    main()
