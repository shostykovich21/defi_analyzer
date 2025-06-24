import yaml
from typing import Dict

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self._validate()
    
    def _validate(self):
        required_sections = ['ethereum', 'clustering', 'gnn']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Configuration error: Missing required section '{section}' in config.yaml")
        if 'etherscan_api_key' not in self.config['ethereum']:
            raise ValueError("Configuration error: Missing 'etherscan_api_key' in 'ethereum' section.")

    @property
    def etherscan_api_key(self) -> str:
        return self.config['ethereum']['etherscan_api_key']

    @property
    def clustering_params(self) -> Dict:
        return self.config['clustering']

    @property
    def gnn_params(self) -> Dict:
        return self.config['gnn']