import asyncio
import aiohttp
import logging
from typing import List, Dict

ETHERSCAN_MAX_RECORDS = 10000
MAX_RETRIES = 3

class EthereumClient:
    def __init__(self, etherscan_api_key: str):
        self.etherscan_api_key = etherscan_api_key
        self.session = None
        self.rate_limit_delay = 1 / 5

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _fetch_all_pages(self, params: Dict) -> List[Dict]:
        all_results = []
        page = 1
        url = "https://api.etherscan.io/api"
        
        while True:
            params['page'] = page
            params['offset'] = 1000
            
            await asyncio.sleep(self.rate_limit_delay)
            
            response_data = None
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.session.get(url, params=params) as response:
                        response.raise_for_status()
                        response_data = await response.json()
                        break
                except aiohttp.ClientError as e:
                    logging.warning(f"HTTP error on attempt {attempt + 1}: {e}. Retrying...")
                    await asyncio.sleep(0.5 * (2 ** attempt))
            
            if response_data is None:
                logging.error("Failed to fetch data after multiple retries.")
                break

            if response_data['status'] == '0':
                break

            results = response_data.get('result', [])
            if not isinstance(results, list) or not results:
                break
            
            all_results.extend(results)
            page += 1
            
            if len(results) < params['offset']:
                break
            
            if len(all_results) >= ETHERSCAN_MAX_RECORDS and params.get('action') == 'txlist':
                logging.warning(f"Reached Etherscan's {ETHERSCAN_MAX_RECORDS}-record limit for address {params.get('address')}. Older transactions may be missing.")
                break
                
        return all_results

    async def get_wallet_transactions(self, address: str) -> List[Dict]:
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 'latest',
            'sort': 'asc',
            'apikey': self.etherscan_api_key
        }
        return await self._fetch_all_pages(params)