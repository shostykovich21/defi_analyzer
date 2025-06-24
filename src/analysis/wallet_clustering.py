import numpy as np
import pandas as pd
import os
import json
import logging
from joblib import dump, load
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from typing import Dict

class WalletClusteringAnalyzer:
    def __init__(self, clustering_params: Dict, results_dir: str = 'results'):
        self.params = clustering_params
        self.results_dir = results_dir
        self.scaler_path = os.path.join(results_dir, 'scaler.joblib')
        self.pca_path = os.path.join(results_dir, 'pca.joblib')
        self.metadata_path = os.path.join(results_dir, 'clustering_metadata.json')
        self.scaler = None
        self.pca = None
        self.clustering_model = None

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = ['value', 'gasPrice', 'gasUsed', 'timeStamp']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df['isError'] = pd.to_numeric(df['isError'], errors='coerce').fillna(0).astype(int)

        out_feats = (
            df.groupby("from")
            .agg(
                sent_transactions=("hash", "count"),
                total_value_eth_sent=("value", lambda x: x.sum() / 1e18),
                avg_gas_price=("gasPrice", "mean"),
                unique_recipients=("to", "nunique"),
                failed_tx_ratio=("isError", "mean"),
            )
            .rename_axis("address")
        )

        in_feats = (
            df.groupby("to")
            .agg(received_transactions=("hash", "count"))
            .rename_axis("address")
        )

        features_df = out_feats.join(in_feats, how="outer").fillna(0).reset_index()
        features_df['total_transactions'] = features_df['sent_transactions'] + features_df['received_transactions']
        
        return features_df

    def _estimate_dbscan_eps(self, data: np.ndarray, min_samples: int) -> float:
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(data)
        distances, _ = nn.kneighbors(data)
        k_distances = np.sort(distances[:, -1])
        
        eps_percentile = self.params.get('eps_percentile', 95)
        estimated_eps = np.percentile(k_distances, eps_percentile)
        
        logging.info(f"Estimated DBSCAN `eps` parameter at {eps_percentile}th percentile: {estimated_eps:.4f}")
        return estimated_eps if estimated_eps > 0 else 0.5

    def _load_or_fit_preprocessors(self, data: np.ndarray):
        if os.path.exists(self.scaler_path) and os.path.exists(self.pca_path):
            logging.info("Loading existing preprocessors from disk.")
            self.scaler = load(self.scaler_path)
            self.pca = load(self.pca_path)
        else:
            logging.info("Fitting new preprocessors and saving to disk.")
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=0.95)
            self.scaler.fit(data)
            self.pca.fit(self.scaler.transform(data))
            dump(self.scaler, self.scaler_path)
            dump(self.pca, self.pca_path)
        
        scaled_data = self.scaler.transform(data)
        pca_data = self.pca.transform(scaled_data)
        return pca_data

    def perform_clustering(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if features_df.empty:
            return features_df

        feature_cols = [col for col in features_df.columns if col != 'address']
        X = features_df[feature_cols].values
        
        min_samples = self.params.get('min_samples', 5)
        if X.shape[0] < min_samples:
            logging.warning(f"Data has fewer than min_samples ({min_samples}); assigning all to cluster 0.")
            features_df['cluster'] = 0
            return features_df
            
        X_pca = self._load_or_fit_preprocessors(X)
        estimated_eps = self._estimate_dbscan_eps(X_pca, min_samples)
        
        self.clustering_model = DBSCAN(eps=estimated_eps, min_samples=min_samples, n_jobs=-1)
        cluster_labels = self.clustering_model.fit_predict(X_pca)
        features_df['cluster'] = cluster_labels
        
        score = None
        if X_pca.shape[0] >= 10 and len(set(cluster_labels)) > 1:
            sample_size = min(100, X_pca.shape[0])
            try:
                score = silhouette_score(X_pca, cluster_labels, sample_size=sample_size, random_state=42)
                logging.info(f"Sampled silhouette score: {score:.3f} (higher is better)")
            except ValueError:
                logging.warning("Could not compute silhouette score (likely only one cluster found).")

        metadata = {
            "dbscan_eps": estimated_eps,
            "dbscan_min_samples": min_samples,
            "silhouette_score": score,
            "num_clusters_found": len(set(cluster_labels) - {-1}),
            "num_noise_points": int((cluster_labels == -1).sum())
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Clustering metadata saved to {self.metadata_path}")

        return features_df