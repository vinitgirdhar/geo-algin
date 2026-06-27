# GEO-ALIGN: Cross-Modal Satellite Image Retrieval & GIS Diagnostics

> **ISRO Hackathon Winning Submission**  
> A unified cross-modal vector alignment engine that maps Sentinel-1 active radar backscatter (SAR) directly into Sentinel-2 optical spectral feature spaces using a non-linear 2-layer Neural MLP Adapter, enabling sub-millisecond cloud-resilient search, semantic change detection, and explainable GIS diagnosis.

---

## 🛰️ The Problem: The Cloud & Sensor Gap
1. **Optical Satellites (Sentinel-2)** capture rich spectral details of crops, urban structures, and water colors, but are **completely blinded by cloud cover, night, storm systems, and smoke**.
2. **Radar Satellites (Sentinel-1 SAR)** pierce through weather and darkness, mapping physical surface texture, but the resulting grayscale backscatter is complex, noisy, and difficult for standard computer vision models to match with optical baselines.
3. **Disaster Management Obstacle:** During floods or storms, optical satellites are useless. Rescue teams require a way to query active radar inputs (S1) and instantly retrieve the corresponding pre-disaster optical maps (S2) to evaluate damage. However, the physical domain gap between microwave backscatter and reflected visible light makes cross-modal lookup extremely difficult.

---

## ⚡ The Solution: GEO-ALIGN
GEO-ALIGN bridges the domain gap by learning a shared embedding space between active microwave radar and visible spectral light.

* **Dual-Modal Feature Extraction:** Sentinel-1 (SAR) and Sentinel-2 (Optical) patches are processed through a ResNet18 backbone to extract 512-dimensional semantic embeddings.
* **Non-Linear Neural Projection Adapter:** A 2-layer Multilayer Perceptron (MLP) with GELU activations, Batch Normalization, and Residual Connections projects the SAR embeddings onto the L2-normalized optical unit hypersphere.
* **Optimization:** Trained on 16,000 co-registered Sentinel image pairs using a hybrid **Cosine Embedding Loss + Mean Squared Error (MSE)** loss, optimized using AdamW and Cosine Annealing learning rate schedules.
* **Retrieval Performance:** Delivers a **5.6x improvement** in geographic instance matching compared to standard linear adapters:
  * **S1-to-S2 Cross-Modal Instance MAP:** **35.8%**
  * **S2-to-S1 Cross-Modal Instance MAP:** **57.1%**
  * **Average Query Retrieval Latency:** **1.16 ms** (CPU)

---

## 🌟 Advanced GIS Diagnostic Features

### 🔍 1. Explainable Retrieval Engine (Grad-CAM)
* Computes the gradient of the cross-modal similarity score (dot product) relative to the activations of the final convolutional layer (`layer4` of ResNet18).
* Renders a Jet color overlay (Red/Yellow = High focus) showing the exact visual structures (e.g. roads, crop boundaries) the model utilized to determine similarity.

### 🗺️ 2. Semantic Change Detection Map
* Analyzes pixel-level deep-feature differences between the co-registered query and database match.
* Classifies changes using active SAR backscatter properties:
  * 🔵 **Flood / Water boundary change:** Low radar backscatter regions ($<0.22$).
  * 🔴 **Urban / Construction change:** High double-bounce radar reflection regions ($>0.65$).
  * 🟢 **Vegetation / Forest cover change:** Medium diffuse backscatter regions.

### 🎲 3. MC Dropout Uncertainty Estimation
* Runs 10 forward passes through the projection head with active dropout layers.
* Maps the embedding variance to a spatial uncertainty heatmap (representing pixel stability) and outputs a calibrated confidence percentage and a reliability tag (`HIGH`/`MEDIUM`/`LOW`).

---

## 🛠️ System Architecture

```mermaid
flowchart LR
    %% Define sleek color styling classes
    classDef s_input fill:#121824,stroke:#46e5ff,stroke-width:2px,color:#fff;
    classDef s_model fill:#251632,stroke:#ff4d9d,stroke-width:2px,color:#fff;
    classDef s_db fill:#091a24,stroke:#00c3ff,stroke-width:2px,color:#fff;
    classDef s_ui fill:#112415,stroke:#46e582,stroke-width:2px,color:#fff;

    %% Data Inputs Section
    subgraph Inputs ["1. Raw Satellite Inputs"]
        S1["Sentinel-1 SAR (Active Radar)"]:::s_input
        S2["Sentinel-2 Optical (Reflected RGB)"]:::s_input
    end

    %% Preprocessing Pipeline
    subgraph Prep ["2. DSP Preprocessing"]
        S1_Prep["Lee Speckle Filter & Log Scaling (dB)"]:::s_input
        S2_Prep["RGB Band Normalization"]:::s_input
    end

    %% Deep Feature Extraction
    subgraph Feat ["3. Feature Extraction"]
        Backbone["Frozen ResNet18 Encoder"]:::s_model
        S1_RawEmbed["SAR Raw Embeddings (512-D)"]:::s_model
        S2_RawEmbed["Optical Embeddings (512-D)"]:::s_model
    end

    %% Cross-Modal Alignment Network
    subgraph Alignment ["4. Neural Cross-Modal Alignment"]
        MLP["2-Layer MLP Projection Head<br/>(GELU + BatchNorm + Residual)"]:::s_model
        S1_Aligned["S1 Aligned Embeddings (512-D)"]:::s_model
        Loss["Cosine Embedding Loss + MSE Regression"]:::s_model
    end

    %% Retrieval Engine
    subgraph Engine ["5. FastAPI Vector Index Engine"]
        DB["Embeddings Cache Index (16,000 Pairs)"]:::s_db
        Matcher["Exact Cosine Similarity Matcher"]:::s_db
        
        subgraph XAI ["Explainable AI & Change Detection"]
            GradCAM["Grad-CAM Attention Map Generator"]:::s_db
            MCDropout["Monte Carlo Dropout Uncertainty Estimator"]:::s_db
            ChangeDetect["Deep Feature Semantic Change Detection"]:::s_db
        end
    end

    %% Client Frontend Dashboard
    subgraph Client ["6. React Workspace UI (Vite + Netlify)"]
        UI_Retrieval["Retrieval Core (Signal Scan & Radar Sweep)"]:::s_ui
        UI_Telemetry["Telemetry Center (Model Diagnostic Console)"]:::s_ui
        UI_Inspector["Interactive Inspector (Swipe Compare / CAM / Change Map / Uncertainty)"]:::s_ui
    end

    %% Flow Connections
    S1 --> S1_Prep
    S2 --> S2_Prep
    
    S1_Prep --> Backbone
    S2_Prep --> Backbone
    
    Backbone --> S1_RawEmbed
    Backbone --> S2_RawEmbed
    
    S1_RawEmbed --> MLP
    MLP --> S1_Aligned
    
    S1_Aligned & S2_RawEmbed -.-> Loss
    S1_Aligned & S2_RawEmbed --> DB
    
    DB --> Matcher
    Matcher --> XAI
    
    XAI --> Client
    UI_Retrieval --> Matcher
    UI_Inspector <-- XAI
    UI_Telemetry <-- DB
```

---

## 💻 Local Installation & Setup

### Prerequisites
* Python 3.9+
* Node.js 16+

### 1. Backend Setup
1. Navigate to the project root:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI backend server:
   ```bash
   python main.py
   ```
   * The API will boot on `http://localhost:8000`.

### 2. Frontend Setup
1. From the project root, install Node packages:
   ```bash
   npm install
   ```
2. Run the Vite development server:
   ```bash
   npm run dev
   ```
   * Open `http://localhost:5173/` in your browser.

---

## 🌐 Production Deployment

### 1. Frontend (Netlify)
The React app is pre-configured with a custom `netlify.toml` file in the root. 
* **Build command:** `npm run build`
* **Publish folder:** `dist`
* **API Configuration:** Set the environment variable `VITE_API_BASE_URL` in your Netlify dashboard to your deployed backend URL.

### 2. Backend (Docker / Cloud Run)
A multi-stage `Dockerfile` is provided in the `backend/` directory to containerize the FastAPI service for deployment on Google Cloud Run, Render, or a VPS.
* **Build Docker Image:**
  ```bash
  docker build -t geo-align-backend ./backend
  ```
