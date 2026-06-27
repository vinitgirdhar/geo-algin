"""
Improved Cross-Modal Alignment using a 2-layer MLP with Cosine Embedding Loss.
This replaces the simple Ridge regression with a non-linear projection head
that directly optimizes for cosine similarity between paired S1→S2 embeddings.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AlignmentMLP(nn.Module):
    """2-layer MLP projection head: 512 → 512 → 512 with residual connection."""
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        return x + self.net(x)  # Residual connection

class RidgeWrapper:
    """Wrapper so the MLP can be used in place of sklearn Ridge in the search index."""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_t = torch.from_numpy(X).float().to(self.device)
            else:
                X_t = X.float().to(self.device)
            out = self.model(X_t)
            return out.cpu().numpy()

def main():
    cache_path = "d:/Cooking stuff/Isro_Hackthon/backend/embeddings_cache.pkl"
    if not os.path.exists(cache_path):
        print(f"Cache not found at {cache_path}")
        return
    
    print("Loading embeddings cache...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    
    s1_features = cache["s1_features"]   # (N, 512)
    s2_features = cache["s2_features"]   # (N, 512)
    metadata = cache["metadata"]
    
    # L2-normalize inputs
    s1_norm = s1_features / (np.linalg.norm(s1_features, axis=1, keepdims=True) + 1e-9)
    s2_norm = s2_features / (np.linalg.norm(s2_features, axis=1, keepdims=True) + 1e-9)
    
    device = 'cpu'
    
    # Convert to tensors
    s1_tensor = torch.from_numpy(s1_norm).float()
    s2_tensor = torch.from_numpy(s2_norm).float()
    
    dataset = TensorDataset(s1_tensor, s2_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)
    
    # Initialize model
    model = AlignmentMLP(dim=512).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    # Combined loss: Cosine similarity + MSE for paired alignment
    cos_loss_fn = nn.CosineEmbeddingLoss(margin=0.0)
    mse_loss_fn = nn.MSELoss()
    
    num_epochs = 80
    print(f"Training AlignmentMLP for {num_epochs} epochs on {len(dataset)} pairs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for s1_batch, s2_batch in dataloader:
            s1_batch = s1_batch.to(device)
            s2_batch = s2_batch.to(device)
            
            # Forward pass
            s1_projected = model(s1_batch)
            
            # Normalize projected output
            s1_proj_norm = nn.functional.normalize(s1_projected, dim=1)
            s2_batch_norm = nn.functional.normalize(s2_batch, dim=1)
            
            # Cosine embedding loss (target=+1 means paired, maximize similarity)
            target = torch.ones(s1_batch.size(0)).to(device)
            loss_cos = cos_loss_fn(s1_proj_norm, s2_batch_norm, target)
            
            # MSE loss for direct regression
            loss_mse = mse_loss_fn(s1_proj_norm, s2_batch_norm)
            
            loss = loss_cos + 0.5 * loss_mse
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Quick eval: compute average cosine similarity on full dataset
            model.eval()
            with torch.no_grad():
                all_projected = model(s1_tensor.to(device))
                all_proj_norm = nn.functional.normalize(all_projected, dim=1)
                s2_full_norm = nn.functional.normalize(s2_tensor.to(device), dim=1)
                cos_sims = (all_proj_norm * s2_full_norm).sum(dim=1)
                avg_cos = cos_sims.mean().item()
                
                # Instance retrieval: check how many paired items land in top-1
                sim_matrix = torch.mm(all_proj_norm, s2_full_norm.T)
                top1_indices = sim_matrix.argmax(dim=1)
                correct = (top1_indices == torch.arange(len(s1_tensor)).to(device)).float()
                top1_acc = correct.mean().item()
                
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Avg CosSim: {avg_cos:.4f} | Top-1 Acc: {top1_acc*100:.1f}%")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        s1_aligned_tensor = model(s1_tensor.to(device))
        s1_aligned_np = s1_aligned_tensor.cpu().numpy()
    
    # Wrap model for compatibility with index_search.py
    ridge_wrapper = RidgeWrapper(model, device)
    
    # Update cache
    cache["s1_aligned_features"] = s1_aligned_np
    cache["alignment_model"] = ridge_wrapper
    
    print("Saving improved embeddings cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("[OK] AlignmentMLP training complete. Cache updated.")

if __name__ == "__main__":
    main()
