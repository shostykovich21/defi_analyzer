# ChainProfile ­– High-Speed DeFi Wallet Analyzer

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-FF69B4.svg)

A lightning fast tool that profiles Ethereum wallets.  
It grabs on-chain transactions, clusters similar addresses, and learns dense graph neural network embeddings.  
Runs end-to-end in seconds on a laptop.

---

## Why It’s Useful

* No slow loops. Pure pandas vectorization.  
* DBSCAN clustering with auto-tuned `eps` and silhouette check.  
* GNN with early stopping to dodge overfitting.  
* All settings live in `config.yaml`.  
* Preprocessors and metadata saved for repeatable runs.

---

## Tech

* **Python 3.8+**  
* pandas, scikit-learn, joblib  
* PyTorch + PyTorch Geometric  
* aiohttp, Web3.py  

---

## Two Quick Scripts

1. **`1_fetch_sample_data.py`** – one-time, online.  
   Downloads transactions via Etherscan and stores them in `results/`.  
2. **`main_fast.py`** – offline, repeatable.  
   Does feature engineering, clustering, GNN in one go.

---
# How to run the project:
## Install

```bash
git clone https://github.com/shostykovich21/defi_analyzer.git
cd defi_analyzer
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate
````

Install PyTorch / PyG with the command shown in the official docs for your OS+CUDA, **then**:

```bash
pip install -r requirements.txt
```

Create `config.yaml`:

```yaml
ethereum:
  etherscan_api_key: "YOUR KEY" #insert your key here
clustering:
  min_samples: 5
  eps_percentile: 95
gnn:
  hidden_dim: 32
  epochs: 50
  patience: 5
  learning_rate: 0.01
```

---

## Run

```bash
# Download data (once)
python 1_fetch_sample_data.py                          
python 1_fetch_sample_data.py --address 0x... --limit 500

# Full pipeline (any time)
python main_fast.py                                    

# To see the slices
python main_fast.py --stage cluster
python main_fast.py --stage gnn
```

---

## Layout

```
defi-wallet-analyzer/
  src/
    data/        # fetch on-chain data
    analysis/    # features + clustering
    models/      # GNN code
    utils/       # config helper
  results/       # CSVs, models, metadata
  1_fetch_sample_data.py
  main_fast.py
  config.yaml
  requirements.txt
```

---

## What You Get in `results/`

* `sample_transactions.csv` – raw Etherscan data
* `fast_wallet_clusters.csv` – features + cluster ids
* `wallet_embeddings_gnn_fast.csv` – learned address vectors
* `scaler.joblib`, `pca.joblib` – saved preprocessors
* `clustering_metadata.json` – eps, silhouette, counts

---

## License

MIT

```
```
