import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict

class WalletGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(WalletGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class AddressClusteringGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_clusters: int):
        super(AddressClusteringGNN, self).__init__()
        self.embedding_gnn = WalletGNN(input_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_clusters)
        
    def forward(self, data: Data):
        embeddings = self.embedding_gnn(data.x, data.edge_index)
        logits = self.classifier(embeddings)
        return embeddings, logits

class GraphLearningAnalyzer:
    def __init__(self, gnn_params: Dict, device: str = None):
        self.params = gnn_params
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = None
        self.node_mapping = {}

    def build_transaction_graph(self, transactions_df: pd.DataFrame, features_df: pd.DataFrame) -> Data:
        features_df = features_df.reset_index(drop=True)
        self.node_mapping = {addr: i for i, addr in enumerate(features_df['address'])}
        self.inverse_node_mapping = {i: addr for addr, i in self.node_mapping.items()}
        
        valid_txs = transactions_df[
            transactions_df['from'].isin(self.node_mapping) &
            transactions_df['to'].isin(self.node_mapping)
        ].copy()

        valid_txs['from_idx'] = valid_txs['from'].map(self.node_mapping)
        valid_txs['to_idx'] = valid_txs['to'].map(self.node_mapping)

        edge_index = torch.tensor(valid_txs[['from_idx', 'to_idx']].values, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(self.node_mapping))
        
        feature_cols = [col for col in features_df.columns if col not in ['address', 'cluster']]
        node_features = torch.tensor(features_df[feature_cols].values, dtype=torch.float)

        graph_data = Data(x=node_features, edge_index=edge_index, num_nodes=len(self.node_mapping))
        graph_data.y = torch.tensor(features_df['cluster'].values, dtype=torch.long)
        
        return graph_data

    def train_model(self, data: Data):
        num_features = data.x.shape[1]
        trainable_mask = data.y != -1
        labels = data.y[trainable_mask]
        
        if labels.numel() == 0 or len(torch.unique(labels)) < 2:
            logging.warning("Not enough valid clusters to train a model. Skipping GNN training.")
            return

        num_clusters = len(torch.unique(labels))
        hidden_dim = self.params.get('hidden_dim', 32)
        epochs = self.params.get('epochs', 50)
        lr = self.params.get('learning_rate', 0.01)
        patience = self.params.get('patience', 5)

        self.model = AddressClusteringGNN(num_features, hidden_dim, num_clusters).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        
        cluster_id_map = {old_id.item(): new_id for new_id, old_id in enumerate(torch.unique(labels))}
        mapped_labels = torch.tensor([cluster_id_map[l.item()] for l in labels], dtype=torch.long)

        trainable_indices = torch.where(trainable_mask)[0].cpu().numpy()
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            trainable_indices, mapped_labels.cpu().numpy(), test_size=0.2, random_state=42, stratify=mapped_labels.cpu().numpy()
        )

        data = data.to(self.device)
        train_labels = torch.tensor(train_labels, dtype=torch.long).to(self.device)
        val_labels = torch.tensor(val_labels, dtype=torch.long).to(self.device)

        best_val_loss = float('inf')
        wait = 0

        logging.info(f"Training GNN for up to {epochs} epochs on device: {self.device} with early stopping...")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            _, logits = self.model(data)
            loss = F.cross_entropy(logits[train_indices], train_labels)
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                _, val_logits = self.model(data)
                val_loss = F.cross_entropy(val_logits[val_indices], val_labels)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logging.info(f"Stopping early at epoch {epoch}. Validation loss did not improve.")
                    break
        
        self.model.eval()

    def get_node_embeddings(self, data: Data) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self.model(data)
        
        embeddings_df = pd.DataFrame(embeddings.cpu().numpy(), columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
        embeddings_df['address'] = embeddings_df.index.map(self.inverse_node_mapping.get)
        
        return embeddings_df[['address'] + [col for col in embeddings_df.columns if col != 'address']]