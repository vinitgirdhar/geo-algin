import os
import glob
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.linear_model import Ridge
from tqdm import tqdm

# Import preprocessing functions
from preprocess import preprocess_s1, preprocess_s2

def main():
    t_start = time.time()
    v2_dir = "d:/Cooking stuff/Isro_Hackthon/v_2"
    categories = ["agri", "barrenland", "grassland", "urban"]
    
    print("Finding Sentinel-1 and Sentinel-2 image pairs...")
    pairs = []
    for cat in categories:
        s1_pattern = os.path.join(v2_dir, cat, "s1", "*.png")
        s1_files = glob.glob(s1_pattern)
        print(f"Category '{cat}': found {len(s1_files)} S1 images.")
        for s1_path in s1_files:
            # Construct corresponding S2 path
            # S1 path: .../cat/s1/ROIs1868_summer_s1_59_p10.png
            # S2 path: .../cat/s2/ROIs1868_summer_s2_59_p10.png
            s2_path = s1_path.replace("\\s1\\", "\\s2\\").replace("/s1/", "/s2/").replace("_s1_", "_s2_")
            if os.path.exists(s2_path):
                pairs.append({
                    'category': cat,
                    's1_path': s1_path,
                    's2_path': s2_path
                })
            else:
                print(f"Warning: Corresponding S2 image not found for {s1_path}")
                
    print(f"Total matched pairs: {len(pairs)}")
    if len(pairs) == 0:
        print("Error: No paired images found!")
        return

    # Load ResNet18 model pretrained on ImageNet
    print("Loading pretrained ResNet18 feature extractor...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final classification layer with Identity to extract 512-dim features
    model.fc = nn.Identity()
    model.eval()

    # Extract S1 features
    print("Extracting S1 features...")
    batch_size = 32
    s1_features_list = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="S1 Extraction"):
        batch_pairs = pairs[i : i + batch_size]
        batch_tensors = []
        for p in batch_pairs:
            img = preprocess_s1(p['s1_path'])
            img_t = torch.from_numpy(img).permute(2, 0, 1)
            batch_tensors.append(img_t)
        batch_tensor = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = model(batch_tensor)
            s1_features_list.append(feats.numpy())
    s1_features = np.concatenate(s1_features_list, axis=0)
    print(f"S1 features shape: {s1_features.shape}")

    # Extract S2 features
    print("Extracting S2 features...")
    s2_features_list = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="S2 Extraction"):
        batch_pairs = pairs[i : i + batch_size]
        batch_tensors = []
        for p in batch_pairs:
            img = preprocess_s2(p['s2_path'])
            img_t = torch.from_numpy(img).permute(2, 0, 1)
            batch_tensors.append(img_t)
        batch_tensor = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = model(batch_tensor)
            s2_features_list.append(feats.numpy())
    s2_features = np.concatenate(s2_features_list, axis=0)
    print(f"S2 features shape: {s2_features.shape}")

    # Train linear mapping using Ridge Regression
    print("Training Ridge regression mapping S1 to S2 space...")
    t_train_start = time.time()
    # We use multi-output Ridge regression
    alignment_model = Ridge(alpha=1.0)
    alignment_model.fit(s1_features, s2_features)
    t_train_end = time.time()
    print(f"Ridge regression training completed in {t_train_end - t_train_start:.4f} seconds.")

    # Align S1 features to S2 space
    print("Aligning S1 features to S2 space...")
    s1_aligned_features = alignment_model.predict(s1_features)
    print(f"Aligned S1 features shape: {s1_aligned_features.shape}")

    # Create metadata dictionary list
    # Convert absolute paths to relative paths or keep them as is. Let's keep them, but also add category.
    metadata = []
    for idx, p in enumerate(pairs):
        metadata.append({
            'index': idx,
            'category': p['category'],
            's1_path': p['s1_path'],
            's2_path': p['s2_path']
        })

    # Cache everything
    cache_path = "d:/Cooking stuff/Isro_Hackthon/backend/embeddings_cache.pkl"
    print(f"Saving embeddings and alignment model to {cache_path}...")
    cache_data = {
        's1_features': s1_features,
        's2_features': s2_features,
        's1_aligned_features': s1_aligned_features,
        'metadata': metadata,
        'alignment_model': alignment_model
    }
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    t_end = time.time()
    print(f"Total training & alignment pipeline completed in {t_end - t_start:.2f} seconds.")

if __name__ == '__main__':
    main()
