from py_clob_client.client import ClobClient

HOST = "https://clob.polymarket.com"

def main():
    client = ClobClient(HOST)  # Level 0 (no auth)
    print("OK:", client.get_ok())
    print("TIME:", client.get_server_time())

if __name__ == "__main__":
    main()
