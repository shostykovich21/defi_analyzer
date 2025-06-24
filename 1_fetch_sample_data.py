import asyncio
import pandas as pd
import os
import logging
import argparse

from src.data.ethereum_client import EthereumClient
from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main(args):
    os.makedirs('results', exist_ok=True)
    
    try:
        config = Config()
    except (FileNotFoundError, ValueError) as e:
        logging.fatal(e)
        return
        
    logging.info(f"Fetching a sample dataset for address: {args.address}")
    
    async with EthereumClient(config.etherscan_api_key) as eth_client:
        
        logging.info("Collecting wallet transactions...")
        transactions = await eth_client.get_wallet_transactions(args.address)
        
        if transactions:
            output_path = 'results/sample_transactions.csv'
            transactions_df = pd.DataFrame(transactions)
            
            if len(transactions_df) > args.limit:
                transactions_df = transactions_df.head(args.limit)

            transactions_df.to_csv(output_path, index=False)
            logging.info(f"Success! Fetched {len(transactions_df)} sample transactions.")
            logging.info(f"Saved to {output_path}. You can now run 'main_fast.py'.")
        else:
            logging.error("Could not fetch sample transactions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch sample transaction data from Etherscan.")
    parser.add_argument('--address', type=str, default="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", help='The Ethereum address to fetch transactions for.')
    parser.add_argument('--limit', type=int, default=1000, help='The maximum number of transactions to save.')
    args = parser.parse_args()
    asyncio.run(main(args))