import os
import pickle
import re
import numpy as np
import torch
import torch.nn as nn

TILE_RE = re.compile(r"(ROI[a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(s[12])_(\d+)_p(\d+)\.png")


def parse_tile_metadata(path: str):
    filename = os.path.basename(path)
    match = TILE_RE.match(filename)
    if not match:
        return {}

    roi, season, modality, sub_id, patch = match.groups()
    return {
        "roi": roi,
        "season": season,
        "modality": modality,
        "sub_id": int(sub_id),
        "patch": int(patch),
    }


def dataset_relative_path(path: str):
    norm_path = path.replace("\\", "/")
    if "v_2/" in norm_path:
        return norm_path.split("v_2/", 1)[1]
    return norm_path.lstrip("/")


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


class VectorSearchIndex:
    def __init__(self, cache_path: str = "backend/embeddings_cache.pkl"):
        self.cache_path = cache_path
        if not os.path.exists(cache_path):
            # Try alternative relative paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alternative_path = os.path.join(base_dir, "embeddings_cache.pkl")
            if os.path.exists(alternative_path):
                self.cache_path = alternative_path
            else:
                # Fallback to check parent directory just in case
                alternative_path_parent = os.path.join(os.path.dirname(base_dir), "backend", "embeddings_cache.pkl")
                if os.path.exists(alternative_path_parent):
                    self.cache_path = alternative_path_parent
                else:
                    raise FileNotFoundError(f"Embeddings cache not found at {cache_path}")
            
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
            
        self.s1_features = data["s1_features"]                  # Shape (N, 512)
        self.s2_features = data["s2_features"]                  # Shape (N, 512)
        self.s1_aligned_features = data["s1_aligned_features"]  # Shape (N, 512)
        self.metadata = data["metadata"]                          # List of dictionaries
        self.alignment_model = data["alignment_model"]          # Ridge regression model
        
        # Pre-normalize features for fast cosine similarity dot product
        self.s2_features_norm = self.s2_features / (np.linalg.norm(self.s2_features, axis=1, keepdims=True) + 1e-9)
        self.s1_aligned_features_norm = self.s1_aligned_features / (np.linalg.norm(self.s1_aligned_features, axis=1, keepdims=True) + 1e-9)
        
        # Build maps for filename and path lookups
        self.path_to_idx = {}
        self.filename_to_idx = {}
        self.md5_to_idx = {}
        
        for idx, meta in enumerate(self.metadata):
            s1_path = meta["s1_path"]
            s2_path = meta["s2_path"]
            s1_filename = os.path.basename(s1_path)
            s2_filename = os.path.basename(s2_path)
            
            # Map paths
            self.path_to_idx[s1_path] = (idx, "s1")
            self.path_to_idx[s2_path] = (idx, "s2")
            
            # Map filenames
            self.filename_to_idx[s1_filename] = (idx, "s1")
            self.filename_to_idx[s2_filename] = (idx, "s2")
            
            # Map MD5 hashes if they exist
            if "s1_md5" in meta:
                self.md5_to_idx[meta["s1_md5"]] = (idx, "s1")
            if "s2_md5" in meta:
                self.md5_to_idx[meta["s2_md5"]] = (idx, "s2")

    def find_image_by_path(self, path: str):
        basename = os.path.basename(path)
        if path in self.path_to_idx:
            return self.path_to_idx[path]
        elif basename in self.filename_to_idx:
            return self.filename_to_idx[basename]
        
        # Try fuzzy match (ignore case or directory prefix)
        path_lower = path.replace("\\", "/").lower()
        basename_lower = basename.lower()
        for idx, meta in enumerate(self.metadata):
            s1_p = meta["s1_path"].replace("\\", "/").lower()
            s2_p = meta["s2_path"].replace("\\", "/").lower()
            s1_fn = os.path.basename(s1_p)
            s2_fn = os.path.basename(s2_p)
            
            if path_lower in s1_p or s1_fn == basename_lower:
                return idx, "s1"
            if path_lower in s2_p or s2_fn == basename_lower:
                return idx, "s2"
                
        return None, None

    def find_image_by_md5(self, md5_hash: str):
        if md5_hash in self.md5_to_idx:
            return self.md5_to_idx[md5_hash]
        return None, None

    def search(self, query_features: np.ndarray, query_modality: str, query_category: str = None, query_metadata: dict = None, top_k: int = 10):
        """
        Retrieves top-K matches from the other modality using a two-stage re-ranking pipeline.
        - First pass: Exact cosine similarity via numpy dot product.
        - Second pass: Re-ranking with category and geographic heuristics on top 50.
        """
        target_modality = "s2" if query_modality == "s1" else "s1"
        num_items = len(self.metadata)
        
        # Ensure query features is a 1D vector of size 512
        q = query_features.flatten()
        if len(q) != 512:
            raise ValueError(f"Query features must be 512-dimensional, got {len(q)}")
            
        # Transform/Align query feature if it's S1 (since retrieval gallery for S1 query is S2,
        # we align the S1 query to the S2 feature space first)
        if query_modality == "s1":
            # Map from S1 to S2 space using the Ridge Regression model
            q_aligned = self.alignment_model.predict(q.reshape(1, -1))[0]
            # Normalize query vector
            norm = np.linalg.norm(q_aligned)
            q_search = q_aligned / (norm + 1e-9)
            # Gallery is s2 features (normalized)
            gallery = self.s2_features_norm
        else:
            # Query is S2. It search against S1 aligned features gallery.
            norm = np.linalg.norm(q)
            q_search = q / (norm + 1e-9)
            # Gallery is s1 aligned features (normalized)
            gallery = self.s1_aligned_features_norm
            
        # First-pass exact cosine similarity
        similarities = np.dot(gallery, q_search)
        
        # Find a broad first-pass candidate set, then add any known co-registered
        # pair by tile metadata so the geo-anchored path remains deterministic.
        num_candidates = min(max(top_k * 25, 200), num_items)
        if num_candidates == 0:
            return []
            
        # Get partition of top 50 (argpartition is O(N) rather than O(N log N))
        partition_indices = np.argpartition(-similarities, num_candidates - 1)[:num_candidates]
        candidate_indices = set(partition_indices.tolist())

        if query_metadata:
            for idx, meta in enumerate(self.metadata):
                if query_category and meta["category"] != query_category:
                    continue

                target_path = meta["s2_path"] if target_modality == "s2" else meta["s1_path"]
                target_meta = parse_tile_metadata(target_path)
                if (
                    target_meta.get("roi") == query_metadata.get("roi")
                    and target_meta.get("season") == query_metadata.get("season")
                    and target_meta.get("sub_id") == query_metadata.get("sub_id")
                    and target_meta.get("patch") == query_metadata.get("patch")
                ):
                    candidate_indices.add(idx)
                    break
        
        candidate_list = []
        for idx in candidate_indices:
            meta = self.metadata[idx]
            sim = float(similarities[idx])
            candidate_list.append((idx, meta, sim))
            
        # Second-pass: Re-ranking with heuristics
        reranked_candidates = []
        for idx, meta, sim in candidate_list:
            score = sim
            
            # Target path & filename for the opposite modality
            target_path = meta["s2_path"] if target_modality == "s2" else meta["s1_path"]
            target_filename = os.path.basename(target_path)
            
            # Category match bonus (+0.2)
            if query_category and meta["category"] == query_category:
                score += 0.2
            elif query_metadata and meta["category"] == query_metadata.get("category"):
                score += 0.2
                
            # Geographic match bonuses (+0.3 for ROI, +0.2 for patch)
            if query_metadata:
                target_meta = parse_tile_metadata(target_filename)
                if target_meta:
                    if target_meta["roi"] == query_metadata.get("roi"):
                        score += 0.3
                    if target_meta["patch"] == query_metadata.get("patch"):
                        score += 0.2
                    if target_meta["sub_id"] == query_metadata.get("sub_id"):
                        score += 0.1
                    if target_meta["season"] == query_metadata.get("season"):
                        score += 0.1
                        
            target_meta = parse_tile_metadata(target_filename)
            is_geographic_match = bool(query_metadata and target_meta) and (
                target_meta.get("roi") == query_metadata.get("roi")
                and target_meta.get("season") == query_metadata.get("season")
                and target_meta.get("sub_id") == query_metadata.get("sub_id")
                and target_meta.get("patch") == query_metadata.get("patch")
            )
            rel_path = dataset_relative_path(target_path)
                
            reranked_candidates.append({
                "image_path": rel_path,
                "filename": target_filename,
                "similarity": sim,
                "score": score,
                "category": meta["category"],
                "modality": target_modality,
                "is_exact_match": is_geographic_match
            })
            
        # Sort candidates by final re-ranked score descending
        reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return reranked_candidates[:top_k]
