import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    computed directly on the ResNet18 model for similarity-based retrieval.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def generate_heatmap(self, x, query_feat):
        self.model.zero_grad()
        
        # Enable gradients for inputs if not already enabled
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        features = self.model(x)  # Shape: (1, 512)
        
        # Compute dot product similarity score
        q_tensor = torch.from_numpy(query_feat).float().to(x.device)
        score = torch.dot(features.squeeze(), q_tensor)
        
        # Backward pass
        score.backward()
        
        # Retrieve gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]      # Shape: (512, H, W)
        activations = self.activations.cpu().data.numpy()[0]  # Shape: (512, H, W)
        
        # Mean gradients over space (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2))  # Shape: (512,)
        
        # Linear combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Rectified Linear Unit (ReLU) to keep positive features
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def apply_colormap_jet(cam):
    """
    Applies a Jet colormap to a 2D float array (0.0 to 1.0) and returns RGB uint8 array.
    Blue (low) -> Cyan -> Green -> Yellow -> Red (high).
    """
    H, W = cam.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    r = np.clip(4 * cam - 2, 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(2 - 4 * cam, 0, 1)
    
    rgb[:, :, 0] = (r * 255).astype(np.uint8)
    rgb[:, :, 1] = (g * 255).astype(np.uint8)
    rgb[:, :, 2] = (b * 255).astype(np.uint8)
    return rgb

def overlay_heatmap_on_image(img_path, cam, alpha=0.5):
    """
    Overlays a Grad-CAM heatmap onto the original image.
    """
    # Load original image and resize to 256x256
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Resize CAM to 256x256 using bilinear interpolation
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    cam_resized = cam_tensor.squeeze().numpy()
    
    # Get colormapped CAM
    cam_rgb = apply_colormap_jet(cam_resized).astype(np.float32) / 255.0
    
    # Blend image and heatmap
    blended = alpha * cam_rgb + (1 - alpha) * img_np
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def compute_mc_dropout_uncertainty(model_wrapper, query_feats, num_runs=10):
    """
    Computes Monte Carlo Dropout uncertainty by running the query multiple times
    with dropout layers forced to active status.
    Returns:
    - uncertainty_score: mean variance across embedding dimensions
    - spatial_uncertainty_map: 2D array showing spatial distribution of uncertainty
    """
    model = model_wrapper.model
    model.train() # Enable training mode to activate dropout layers
    
    # Ensure Batch Normalization layers stay in evaluation mode
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
            
    q_tensor = torch.from_numpy(query_feats).float().unsqueeze(0)
    
    outputs = []
    with torch.no_grad():
        for _ in range(num_runs):
            out = model(q_tensor)
            outputs.append(out.squeeze(0).numpy())
            
    outputs = np.array(outputs) # Shape: (num_runs, 512)
    
    # Variance per dimension
    variances = np.var(outputs, axis=0) # Shape: (512,)
    mean_variance = float(np.mean(variances))
    
    # Generate spatial uncertainty weights
    # Normalize variances to act as channel weights
    weights = variances / (np.sum(variances) + 1e-9)
    
    # We can return the calibrated uncertainty score
    # Normalize uncertainty score to a 0.0 - 1.0 range
    calibrated_uncertainty = float(np.clip(mean_variance * 5.0, 0.0, 1.0))
    
    return calibrated_uncertainty, weights

def generate_spatial_uncertainty_heatmap(img_path, resnet_model, channel_weights):
    """
    Projects MC Dropout channel variance weights back to the final convolutional
    layer of ResNet18 to display which areas of the image are most uncertain.
    """
    from preprocess import preprocess_s1, preprocess_s2
    is_s1 = "s1" in img_path.replace("\\", "/").split("/")[-2]
    
    if is_s1:
        img_np = preprocess_s1(img_path)
    else:
        img_np = preprocess_s2(img_path)
        
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    # Get layer4 activations
    activations = []
    def hook(module, input, output):
        activations.append(output)
        
    hook_handle = resnet_model.layer4.register_forward_hook(hook)
    with torch.no_grad():
        _ = resnet_model(img_t)
    hook_handle.remove()
    
    act = activations[0].squeeze(0).numpy() # Shape: (512, H, W)
    
    # Compute weighted sum of activations using uncertainty channel weights
    uncertainty_map = np.zeros(act.shape[1:], dtype=np.float32)
    for i in range(len(channel_weights)):
        uncertainty_map += channel_weights[i] * np.abs(act[i])
        
    # Normalize
    u_min, u_max = np.min(uncertainty_map), np.max(uncertainty_map)
    if u_max > u_min:
        uncertainty_map = (uncertainty_map - u_min) / (u_max - u_min)
    else:
        uncertainty_map = np.zeros_like(uncertainty_map)
        
    # Resize and blend (Jet theme representing uncertainty)
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    cam_tensor = torch.from_numpy(uncertainty_map).unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    u_resized = cam_tensor.squeeze().numpy()
    
    u_rgb = apply_colormap_jet(u_resized).astype(np.float32) / 255.0
    blended = 0.5 * u_rgb + 0.5 * img_np
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def generate_semantic_change_map(query_path, match_path):
    """
    Computes a physical semantic change detection map between a query image
    and the matched image using backscatter/reflectance delta analysis.
    Highlights:
    - Blue: Flood water extent (low microwave backscatter in S1 query but high S2 differences)
    - Red: Infrastructure destruction / new construction (high backscatter changes)
    - Green: Deforestation / vegetative changes (albedo changes)
    """
    # Load and resize to 256x256
    q_img = Image.open(query_path).convert('L').resize((256, 256))
    m_img = Image.open(match_path).convert('L').resize((256, 256))
    
    q_np = np.array(q_img, dtype=np.float32) / 255.0
    m_np = np.array(m_img, dtype=np.float32) / 255.0
    
    # Structural Difference
    diff = np.abs(q_np - m_np)
    
    # Create RGB change map overlay
    change_overlay = np.zeros((256, 256, 4), dtype=np.uint8) # RGBA
    
    # Physical thresholds
    # Water has very low backscatter (typically < 0.20 in SAR intensity)
    is_s1_query = "s1" in query_path.replace("\\", "/").split("/")[-2]
    
    for y in range(256):
        for x in range(256):
            d_val = diff[y, x]
            if d_val > 0.18: # Significant change threshold
                if is_s1_query:
                    sar_val = q_np[y, x]
                else:
                    sar_val = m_np[y, x]
                    
                # Classify based on physical radar reflections
                if sar_val < 0.22:
                    # Flood / Water boundaries (Blue)
                    change_overlay[y, x] = [0, 195, 255, 180] # Semi-transparent Cyan/Blue
                elif sar_val > 0.65:
                    # High backscatter: Building structures, steel, concrete (Red)
                    change_overlay[y, x] = [255, 77, 157, 180] # Semi-transparent Magenta/Red
                else:
                    # Medium backscatter: Vegetation, crop rows (Green)
                    change_overlay[y, x] = [70, 229, 130, 180] # Semi-transparent Green
                    
    # Blend overlay with original query image (as base)
    base_img = Image.open(query_path).convert('RGB').resize((256, 256))
    base_np = np.array(base_img, dtype=np.float32)
    
    for y in range(256):
        for x in range(256):
            alpha = change_overlay[y, x, 3] / 255.0
            if alpha > 0:
                base_np[y, x] = (1 - alpha) * base_np[y, x] + alpha * change_overlay[y, x, :3]
                
    return Image.fromarray(base_np.astype(np.uint8))

def image_to_base64(img):
    """
    Helper to convert a PIL Image into a base64 encoded PNG string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
