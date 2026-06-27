import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge

cache_path = "backend/embeddings_cache.pkl"
if not os.path.exists(cache_path):
    # Try alternative relative path
    cache_path = "d:/Cooking stuff/Isro_Hackthon/backend/embeddings_cache.pkl"

print("Loading cache...")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

s1_features = cache["s1_features"]
s2_features = cache["s2_features"]

print("L2 normalizing features...")
s1_norm = s1_features / (np.linalg.norm(s1_features, axis=1, keepdims=True) + 1e-9)
s2_norm = s2_features / (np.linalg.norm(s2_features, axis=1, keepdims=True) + 1e-9)

print("Training pre-normalized Ridge regression...")
model = Ridge(alpha=1.0)
model.fit(s1_norm, s2_norm)

print("Computing aligned S1 features...")
s1_aligned = model.predict(s1_norm)

# Update cache
cache["s1_aligned_features"] = s1_aligned
cache["alignment_model"] = model

print("Saving updated cache...")
with open(cache_path, "wb") as f:
    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Alignment model successfully upgraded to L2 pre-normalized Ridge!")
