import torch
import os

# --- 1. COMPUTING ENGINE (Hardware Acceleration) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. RESEARCH FEATURE SET (The 15 Pillars) ---
FEATURES = [
    'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 
    'orig_in_degree', 'orig_out_degree', 'orig_pagerank', 'orig_clustering',
    'dest_in_degree', 'dest_out_degree', 'dest_pagerank', 'orig_flow_ratio'
]

# --- 3. MODEL HYPERPARAMETERS (The Brain Logic) ---
MODEL_CONFIG = {
    "input_size": len(FEATURES),
    "hidden_size": 128,
    "num_layers": 2,
    "output_size": 1,  
    "dropout": 0.2     
}

TRAINING_CONFIG = {
    "batch_size": 2048,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "threshold": 0.5   
}

# --- 4. DYNAMIC PATH MANAGEMENT (The Root Logic) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) 

PATHS = {
    "model_save": os.path.join(PROJECT_ROOT, "models", "hybrid_research_model.pth"),
    "data_processed": os.path.join(PROJECT_ROOT, "data", "processed_research_data.csv"),
    "logs": os.path.join(PROJECT_ROOT, "results", "training_logs.csv")
}

for path in [os.path.join(PROJECT_ROOT, "models"), os.path.join(PROJECT_ROOT, "data")]:
    os.makedirs(path, exist_ok=True)

# --- 5. LOGGING STATUS ---
print(f"System Status: Running on [{DEVICE}]")
print(f"Features Loaded: {MODEL_CONFIG['input_size']} dimensions")
