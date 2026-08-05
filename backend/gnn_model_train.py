"""
gnn_model.py  (updated — saves scaler alongside the model)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(current_dir)
data_path   = os.path.join(current_dir, 'data', 'gnn_training_data.csv')
model_dir   = os.path.join(current_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

df     = pd.read_csv(data_path)
X      = df[['elevation', 'slope', 'dist_to_river', 'rainfall']].values
y      = df['label'].values
coords = df[['longitude', 'latitude']].values

print("Building spatial graph …")
adj        = kneighbors_graph(coords, n_neighbors=5, mode='connectivity', include_self=True)
adj_tensor = torch.FloatTensor(adj.toarray())

scaler  = StandardScaler()
X_scaled= scaler.fit_transform(X)

# ── SAVE SCALER HERE ───────────────────────────────────────────────────────
joblib.dump(scaler, os.path.join(model_dir, 'scaler_gnn.pkl'))
print("GNN scaler saved to models/")

X_tensor = torch.FloatTensor(X_scaled.astype(np.float32))
y_tensor = torch.FloatTensor(y).view(-1, 1)


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.projection = nn.Linear(in_f, out_f)
    def forward(self, x, adj):
        return torch.matmul(adj, self.projection(x))


class FloodGNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.gcn1 = GCNLayer(n_feat, 32)
        self.gcn2 = GCNLayer(32, 16)
        self.fc   = nn.Linear(16, 1)
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn2(x, adj))
        return torch.sigmoid(self.fc(x))


model     = FloodGNN(n_feat=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("Training GNN …")
model.train()
for epoch in range(101):
    optimizer.zero_grad()
    output = model(X_tensor, adj_tensor)
    loss   = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        acc = ((output > 0.5).float() == y_tensor).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {acc.item():.4f}")

save_path = os.path.join(model_dir, 'flood_gnn_model.pth')
torch.save(model.state_dict(), save_path)
print(f"GNN model saved to {save_path}")
