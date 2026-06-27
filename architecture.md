# System Architecture Diagram

This file contains the system architecture diagram defined in Mermaid.js syntax. You can view it using any Markdown viewer or by pasting the code into [mermaid.live](https://mermaid.live).

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
