from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

app = FastAPI()

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL ARCHITECTURES ---

class DLinear(nn.Module):
    def __init__(self, seq_len=24, pred_len=1):
        super(DLinear, self).__init__()
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features] -> We take the last feature (load)
        seasonal = self.Linear_Seasonal(x[:, :, -1]) 
        trend = self.Linear_Trend(x[:, :, -1])
        return (seasonal + trend)

class BiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(out)

class Informer(nn.Module):
    def __init__(self, input_size=6, d_model=64, d_ff=256): # d_ff set to 256 to match your weights
        super(Informer, self).__init__()
        self.encoder_input = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 24, d_model))
        
        # Corrected Transformer Layer with explicit d_ff
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=d_ff, # Fixed the 2048 vs 256 mismatch
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder_input(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.decoder(x[:, -1, :])

class FEDformer(nn.Module):
    def __init__(self, input_size=6, d_model=64, n_modes=12):
        super(FEDformer, self).__init__()
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 24, d_model))
        self.w1 = nn.Parameter(torch.randn(d_model, d_model, n_modes, 2))
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, C = x.shape
        x = self.enc_embedding(x) + self.pos_encoder
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros(B, L//2 + 1, 64, device=x.device, dtype=torch.complex64)
        weights = torch.view_as_complex(self.w1)
        out_ft[:, :12, :] = torch.einsum('bjc,cdj->bjd', x_ft[:, :12, :], weights)
        x = torch.fft.irfft(out_ft, n=L, dim=1)
        return self.decoder(x[:, -1, :])

# --- ASSET LOADING ---
models_data = {}
model_classes = {
    "dlinear": DLinear,
    "bilstm": BiLSTM,
    "informer": Informer,
    "fedformer": FEDformer
}

@app.on_event("startup")
def load_assets():
    for name, m_class in model_classes.items():
        try:
            pth_path = f"{name}_model.pth"
            pkl_path = f"{name}_scaler.pkl"
            
            if os.path.exists(pth_path) and os.path.exists(pkl_path):
                model = m_class()
                # Load with strict=False to handle minor metadata differences
                model.load_state_dict(torch.load(pth_path, map_location='cpu'), strict=False)
                model.eval()
                with open(pkl_path, "rb") as f:
                    scaler = pickle.load(f)
                models_data[name] = {"model": model, "scaler": scaler}
                print(f"✅ {name.upper()} loaded successfully.")
            else:
                print(f"⚠️ Files missing for {name}: check .pth and .pkl naming.")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

# --- SCHEMAS ---
class PredictRequest(BaseModel):
    temp: float
    prev_load: float
    isHoliday: int
    month: int
    hour: int
    model_name: str

# --- CORE INFERENCE FUNCTION ---
def run_inference(data: PredictRequest):
    m_key = data.model_name.lower()
    if m_key not in models_data:
        return None

    model = models_data[m_key]["model"]
    scaler = models_data[m_key]["scaler"]

    # Feature order: [temp, prev_load, isHoliday, month, hour, curr_load]
    # We use prev_load as a placeholder for curr_load during the scaling step
    raw_row = [data.temp, data.prev_load, data.isHoliday, data.month, data.hour, data.prev_load]
    seq = np.tile(raw_row, (24, 1))

    # Scale the sequence
    scaled_seq = scaler.transform(seq)
    input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_scaled = output.detach().cpu().numpy().flatten()[0]

    # Inverse Transform Logic
    # We must put the prediction in the 6th column (index 5) of a dummy array
    dummy = np.zeros((1, 6))
    dummy[0, :5] = scaled_seq[-1, :5] # Keep scaled inputs for context
    dummy[0, 5] = pred_scaled         # Put prediction in target column
    
    final_val = scaler.inverse_transform(dummy)[0, 5]

    # Clip to 650 kW Grid Limit for IIT BHU
    return round(float(np.clip(final_val, 0, 650)), 2)

@app.post("/predict")
async def predict(data: PredictRequest):
    result = run_inference(data)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Model '{data.model_name}' not found or failed to load.")
    return {"predicted_load_mw": result, "status": "success"}

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "loaded_models": list(models_data.keys()),
        "expected_models": list(model_classes.keys())
    }

@app.get("/")
def home():
    return {"message": "IIT BHU Power Forecasting API is Live", "docs": "/docs"}
