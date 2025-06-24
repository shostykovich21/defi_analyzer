import pandas as pd
import torch
import os
import time
import logging
import argparse

from src.utils.config import Config
from src.analysis.wallet_clustering import WalletClusteringAnalyzer
from src.models.graph_neural_network import GraphLearningAnalyzer

def run_clustering_stage(config: Config):
    logging.info("--- Running Stage 1: Behavioral Feature Engineering & Clustering ---")
    
    try:
        transactions_df = pd.read_csv('results/sample_transactions.csv', low_memory=False)
    except FileNotFoundError:
        logging.critical("'results/sample_transactions.csv' not found. Please run '1_fetch_sample_data.py' first.")
        return None, None

    clustering_analyzer = WalletClusteringAnalyzer(
        clustering_params=config.clustering_params,
        results_dir='results'
    )
    
    logging.info("Step 1.1: Extracting behavioral features...")
    features_df = clustering_analyzer.extract_behavioral_features(transactions_df)
    
    logging.info("Step 1.2: Performing data-driven clustering...")
    clustered_features_df = clustering_analyzer.perform_clustering(features_df)
    
    clustered_features_df.to_csv('results/fast_wallet_clusters.csv', index=False)
    logging.info("Clustering results saved to 'results/fast_wallet_clusters.csv'.")
    return transactions_df, clustered_features_df

def run_gnn_stage(config: Config, transactions_df: pd.DataFrame = None, clustered_features_df: pd.DataFrame = None):
    logging.info("--- Running Stage 2: Graph Neural Network Analysis ---")

    if transactions_df is None or clustered_features_df is None:
        logging.info("Loading data from previous stage...")
        try:
            transactions_df = pd.read_csv('results/sample_transactions.csv', low_memory=False)
            clustered_features_df = pd.read_csv('results/fast_wallet_clusters.csv', low_memory=False)
        except FileNotFoundError:
            logging.critical("Required files not found. Please run the clustering stage first.")
            return

    graph_analyzer = GraphLearningAnalyzer(gnn_params=config.gnn_params)

    logging.info("Step 2.1: Building transaction graph...")
    graph_data = graph_analyzer.build_transaction_graph(transactions_df, clustered_features_df)
    logging.info(f"Graph created with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges.")

    logging.info("Step 2.2: Training GNN model...")
    graph_analyzer.train_model(graph_data)

    logging.info("Step 2.3: Generating node embeddings...")
    node_embeddings_df = graph_analyzer.get_node_embeddings(graph_data)
    
    if not node_embeddings_df.empty:
        output_path = 'results/wallet_embeddings_gnn_fast.csv'
        node_embeddings_df.to_csv(output_path, index=False)
        logging.info(f"GNN analysis complete. Embeddings saved to {output_path}")
    else:
        logging.warning("GNN model was not trained, so no embeddings were generated.")

def main():
    parser = argparse.ArgumentParser(description="Run the DeFi Wallet Analysis Pipeline.")
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'cluster', 'gnn'], help='Which stage of the pipeline to run.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    logging.info("Starting FAST DeFi Wallet Analysis Pipeline...")

    try:
        config = Config()
    except (FileNotFoundError, ValueError) as e:
        logging.fatal(e)
        return

    transactions_df, clustered_features_df = None, None
    if args.stage in ['all', 'cluster']:
        transactions_df, clustered_features_df = run_clustering_stage(config)
    
    if args.stage in ['all', 'gnn']:
        if transactions_df is not None and clustered_features_df is not None:
            run_gnn_stage(config, transactions_df, clustered_features_df)
        else:
            run_gnn_stage(config)

    end_time = time.time()
    logging.info(f"\nPipeline stage(s) completed in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    main()