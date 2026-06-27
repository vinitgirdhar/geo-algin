import os
import pickle
import json
import time
import numpy as np
import torch
import torch.nn as nn

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
        return x + self.net(x)

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

import sys
sys.modules['__main__'].RidgeWrapper = RidgeWrapper
sys.modules['__main__'].AlignmentMLP = AlignmentMLP


def compute_cosine_similarity(Q, K):
    """
    Computes cosine similarity matrix between queries Q and gallery K.
    """
    # Normalize rows
    Q_norm = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    K_norm = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    # Dot product
    return np.dot(Q_norm, K_norm.T)

def evaluate_retrieval(queries, gallery, query_cats, gallery_cats, mode='cross'):
    """
    Evaluates retrieval and returns a dictionary of metrics:
    - Instance-level metrics: F1@5, F1@10, MAP (only applicable if query index corresponds to gallery index)
    - Category-level metrics: F1@5, F1@10, MAP
    """
    num_queries = len(queries)
    sim_matrix = compute_cosine_similarity(queries, gallery)
    
    # Pre-calculate category counts in gallery
    cats_unique, cats_counts = np.unique(gallery_cats, return_counts=True)
    cat_counts_dict = dict(zip(cats_unique, cats_counts))

    # Metric accumulators
    inst_hits_at_5 = 0
    inst_hits_at_10 = 0
    inst_ap = []

    cat_p_at_5 = []
    cat_r_at_5 = []
    cat_hit_at_5 = []
    cat_p_at_10 = []
    cat_r_at_10 = []
    cat_hit_at_10 = []
    cat_ap = []

    for i in range(num_queries):
        sims = sim_matrix[i].copy()
        
        # If same-modal retrieval, exclude the query itself by setting similarity to -inf
        if mode == 'same':
            sims[i] = -np.inf
            
        # Sort indices in descending order of similarity
        sorted_idx = np.argsort(-sims)
        
        # --- Instance-level evaluation ---
        # The correct instance is the query index i itself
        rank_inst = np.where(sorted_idx == i)[0][0] + 1
        inst_ap.append(1.0 / rank_inst)
        if rank_inst <= 5:
            inst_hits_at_5 += 1
        if rank_inst <= 10:
            inst_hits_at_10 += 1

        # --- Category-level evaluation ---
        q_cat = query_cats[i]
        gallery_hits = (gallery_cats[sorted_idx] == q_cat)
        
        # Total relevant items in the gallery
        n_relevant = cat_counts_dict[q_cat]
        if mode == 'same':
            n_relevant -= 1 # Exclude the query itself
            
        # Top-5 and Top-10 hits
        hits_at_5 = np.sum(gallery_hits[:5])
        hits_at_10 = np.sum(gallery_hits[:10])
        
        # Precision and Recall at 5
        p_5 = hits_at_5 / 5.0
        r_5 = hits_at_5 / n_relevant if n_relevant > 0 else 0.0
        cat_p_at_5.append(p_5)
        cat_r_at_5.append(r_5)
        cat_hit_at_5.append(1.0 if hits_at_5 > 0 else 0.0)

        # Precision and Recall at 10
        p_10 = hits_at_10 / 10.0
        r_10 = hits_at_10 / n_relevant if n_relevant > 0 else 0.0
        cat_p_at_10.append(p_10)
        cat_r_at_10.append(r_10)
        cat_hit_at_10.append(1.0 if hits_at_10 > 0 else 0.0)

        # Average Precision (AP) for category-level
        # AP = sum(Precision@j * relevance[j]) / n_relevant
        # We can compute AP over all gallery items or a threshold. Let's compute full category AP.
        hit_ranks = np.where(gallery_hits)[0] + 1
        precisions_at_hits = np.arange(1, len(hit_ranks) + 1) / hit_ranks
        ap = np.sum(precisions_at_hits) / n_relevant if n_relevant > 0 else 0.0
        cat_ap.append(ap)

    # Compute F1 from average Precision and Recall
    avg_p_5 = np.mean(cat_p_at_5)
    avg_r_5 = np.mean(cat_r_at_5)
    cat_f1_5 = 2.0 * avg_p_5 * avg_r_5 / (avg_p_5 + avg_r_5) if (avg_p_5 + avg_r_5) > 0 else 0.0

    avg_p_10 = np.mean(cat_p_at_10)
    avg_r_10 = np.mean(cat_r_at_10)
    cat_f1_10 = 2.0 * avg_p_10 * avg_r_10 / (avg_p_10 + avg_r_10) if (avg_p_10 + avg_r_10) > 0 else 0.0

    # Instance F1 calculation:
    # Precision@K = hits/K, Recall@K = hits/1
    # Mean Precision@5 = (inst_hits_at_5 / num_queries) / 5
    # Mean Recall@5 = (inst_hits_at_5 / num_queries)
    avg_inst_p_5 = (inst_hits_at_5 / num_queries) / 5.0
    avg_inst_r_5 = (inst_hits_at_5 / num_queries)
    inst_f1_5 = 2.0 * avg_inst_p_5 * avg_inst_r_5 / (avg_inst_p_5 + avg_inst_r_5) if (avg_inst_p_5 + avg_inst_r_5) > 0 else 0.0

    avg_inst_p_10 = (inst_hits_at_10 / num_queries) / 10.0
    avg_inst_r_10 = (inst_hits_at_10 / num_queries)
    inst_f1_10 = 2.0 * avg_inst_p_10 * avg_inst_r_10 / (avg_inst_p_10 + avg_inst_r_10) if (avg_inst_p_10 + avg_inst_r_10) > 0 else 0.0

    results = {
        'instance': {
            'f1_at_5': float(inst_f1_5),
            'f1_at_10': float(inst_f1_10),
            'map': float(np.mean(inst_ap))
        },
        'category': {
            'f1_at_5': float(cat_f1_5),
            'f1_at_10': float(cat_f1_10),
            'precision_at_5': float(avg_p_5),
            'precision_at_10': float(avg_p_10),
            'recall_at_5': float(avg_r_5),
            'recall_at_10': float(avg_r_10),
            'hit_rate_at_5': float(np.mean(cat_hit_at_5)),
            'hit_rate_at_10': float(np.mean(cat_hit_at_10)),
            'map': float(np.mean(cat_ap))
        }
    }
    return results

def main():
    cache_path = "d:/Cooking stuff/Isro_Hackthon/backend/embeddings_cache.pkl"
    if not os.path.exists(cache_path):
        print(f"Error: Embeddings cache not found at {cache_path}. Please run train_align.py first.")
        return

    print("Loading embeddings cache...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    s1_features = cache['s1_features']
    s2_features = cache['s2_features']
    s1_aligned = cache['s1_aligned_features']
    metadata = cache['metadata']

    # Extract categories
    cats = np.array([m['category'] for m in metadata])
    
    print("\n--- Same-Modal Retrieval Evaluation (Category-level) ---")
    s1_to_s1_res = evaluate_retrieval(s1_features, s1_features, cats, cats, mode='same')
    s2_to_s2_res = evaluate_retrieval(s2_features, s2_features, cats, cats, mode='same')
    
    print(f"S1-to-S1 F1@5: {s1_to_s1_res['category']['f1_at_5']:.4f} | F1@10: {s1_to_s1_res['category']['f1_at_10']:.4f} | MAP: {s1_to_s1_res['category']['map']:.4f}")
    print(f"S2-to-S2 F1@5: {s2_to_s2_res['category']['f1_at_5']:.4f} | F1@10: {s2_to_s2_res['category']['f1_at_10']:.4f} | MAP: {s2_to_s2_res['category']['map']:.4f}")

    print("\n--- Cross-Modal Retrieval Evaluation (S1 Aligned -> S2) ---")
    s1_to_s2_res = evaluate_retrieval(s1_aligned, s2_features, cats, cats, mode='cross')
    print("Instance-level:")
    print(f"  F1@5: {s1_to_s2_res['instance']['f1_at_5']:.4f} | F1@10: {s1_to_s2_res['instance']['f1_at_10']:.4f} | MAP: {s1_to_s2_res['instance']['map']:.4f}")
    print("Category-level:")
    print(f"  F1@5: {s1_to_s2_res['category']['f1_at_5']:.4f} | F1@10: {s1_to_s2_res['category']['f1_at_10']:.4f} | MAP: {s1_to_s2_res['category']['map']:.4f}")

    print("\n--- Cross-Modal Retrieval Evaluation (S2 -> S1 Aligned) ---")
    s2_to_s1_res = evaluate_retrieval(s2_features, s1_aligned, cats, cats, mode='cross')
    print("Instance-level:")
    print(f"  F1@5: {s2_to_s1_res['instance']['f1_at_5']:.4f} | F1@10: {s2_to_s1_res['instance']['f1_at_10']:.4f} | MAP: {s2_to_s1_res['instance']['map']:.4f}")
    print("Category-level:")
    print(f"  F1@5: {s2_to_s1_res['category']['f1_at_5']:.4f} | F1@10: {s2_to_s1_res['category']['f1_at_10']:.4f} | MAP: {s2_to_s1_res['category']['map']:.4f}")

    # Measure average query execution time (cross-modal S1-to-S2 similarity search)
    print("\nMeasuring query execution time...")
    t0 = time.time()
    # Perform retrieval for a batch of 100 queries
    num_test_queries = 100
    queries_test = s1_aligned[:num_test_queries]
    sims = compute_cosine_similarity(queries_test, s2_features)
    # Retrieve top 10 for each
    _ = np.argsort(-sims, axis=1)[:, :10]
    t1 = time.time()
    
    avg_query_time = ((t1 - t0) / num_test_queries) * 1000.0 # in milliseconds
    print(f"Average query execution time (1 S1-to-S2 retrieval query): {avg_query_time:.4f} ms")

    # Combine metrics into final JSON format
    metrics_data = {
        "same_modal": {
            "s1_to_s1": {
                "f1_at_5": s1_to_s1_res['category']['f1_at_5'],
                "f1_at_10": s1_to_s1_res['category']['f1_at_10'],
                "map": s1_to_s1_res['category']['map']
            },
            "s2_to_s2": {
                "f1_at_5": s2_to_s2_res['category']['f1_at_5'],
                "f1_at_10": s2_to_s2_res['category']['f1_at_10'],
                "map": s2_to_s2_res['category']['map']
            }
        },
        "cross_modal": {
            "s1_to_s2": {
                "instance": s1_to_s2_res['instance'],
                "category": s1_to_s2_res['category']
            },
            "s2_to_s1": {
                "instance": s2_to_s1_res['instance'],
                "category": s2_to_s1_res['category']
            }
        },
        "performance": {
            "avg_query_time_ms": avg_query_time
        }
    }

    metrics_path = "d:/Cooking stuff/Isro_Hackthon/backend/metrics.json"
    print(f"\nSaving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
        
    print("Evaluation completed successfully.")

if __name__ == '__main__':
    main()
