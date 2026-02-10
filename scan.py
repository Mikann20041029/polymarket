import os
from polymarket_clob_client.client import ClobClient

# 必須Secrets（もう入っている前提）
API_KEY = os.environ["PM_API_KEY"]
API_SECRET = os.environ["PM_API_SECRET"]
PASSPHRASE = os.environ["PM_API_PASSPHRASE"]

# Polymarket mainnet
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon

# まずは“確実に存在する”テスト用 token_id（後で自動取得に戻す）
TEST_TOKEN_ID = "0x0000000000000000000000000000000000000000"  # ← 次で差し替える

def main():
    client = ClobClient(
        host=HOST,
        key=API_KEY,
        secret=API_SECRET,
        passphrase=PASSPHRASE,
        chain_id=CHAIN_ID,
    )

    # 価格取得（BUY側）
    price = client.get_price(token_id=TEST_TOKEN_ID, side="BUY")
    print("PRICE:", price)

if __name__ == "__main__":
    main()
