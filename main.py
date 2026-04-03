from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

app = FastAPI(title="Power Load Forecasting API")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODELS =================

# 1. Corrected DLinear Components
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, back], dim=1).transpose(1, 2)
        x = self.avg(x).transpose(1, 2)
        return x

class DLinear(nn.Module):
    def __init__(self, seq_len=24, pred_len=1, enc_in=6):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Matches your Colab training exactly
        self.moving_avg = MovingAvg(kernel_size=25, stride=1) 
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        trend_init = self.moving_avg(x)
        seasonal_init = x - trend_init

        # Matrix transpositions to match training dimensions
        seasonal_output = self.Linear_Seasonal(seasonal_init.transpose(1, 2))
        trend_output = self.Linear_Trend(trend_init.transpose(1, 2))

        return (seasonal_output + trend_output).transpose(1, 2)

# 2. Other Models (Kept from your original code)
class BiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(out)

class Informer(nn.Module):
    def __init__(self, input_size=6, d_model=64, d_ff=256):
        super(Informer, self).__init__()
        self.encoder_input = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 24, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_ff, batch_first=True)
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

# ================= LOAD MODELS & ASSETS =================

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
            pth = f"{name}_model.pth"
            pkl = f"scaler.pkl" # Assuming all models use the same scaler.pkl

            if os.path.exists(pth) and os.path.exists(pkl):
                model = m_class()
                
                # 🔥 STRICT=TRUE ensures architectures match perfectly
                is_strict = True if name == "dlinear" else False 
                model.load_state_dict(torch.load(pth, map_location="cpu"), strict=is_strict)
                model.eval()

                with open(pkl, "rb") as f:
                    scaler = pickle.load(f)

                models_data[name] = {"model": model, "scaler": scaler}
                print(f"✅ {name.upper()} loaded successfully")
            else:
                print(f"⚠️ Missing files for {name}. Checked for {pth} and {pkl}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

# ================= INPUT SCHEMA =================

class PredictRequest(BaseModel):
    temp: float
    prev_load: float
    isHoliday: int
    month: int
    hour: int
    model_name: str

# ================= HISTORY GENERATION =================

def generate_history_sequence(data):
    """
    NOTE: In a production environment, this should ideally be replaced 
    by fetching the actual last 24 hours of real data from a database.
    """
    base = data.prev_load
    hour = data.hour
    seq = []

    for i in range(24):
        h = (hour - 23 + i) % 24

        if 0 <= h < 5:      factor = 0.55
        elif 5 <= h < 9:    factor = 0.75
        elif 9 <= h < 17:   factor = 0.90
        elif 17 <= h < 22:  factor = 1.15
        else:               factor = 0.80

        hist_load = base * factor

        seq.append([
            data.temp,
            hist_load,
            data.isHoliday,
            data.month,
            h,
            hist_load
        ])

    return np.array(seq)

# ================= INFERENCE LOGIC =================

def run_inference(data: PredictRequest):
    key = data.model_name.lower()
    if key not in models_data:
        return None

    model = models_data[key]["model"]
    scaler = models_data[key]["scaler"]

    # 1. Generate 24h history (Shape: [24, 6])
    seq = generate_history_sequence(data)

    # 2. Scale the input data
    scaled_seq = scaler.transform(seq)
    
    # 3. Convert to Tensor and add Batch Dimension (Shape: [1, 24, 6])
    input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

    # 4. Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 🔥 EXTRACT CORRECT VALUE: 
        # DLinear returns [batch_size, pred_len, features] -> [1, 1, 6]
        # We need the 1st batch, 1st prediction, last feature (-1)
        pred_scaled = outputs[0, 0, -1].cpu().item()

    # 5. Inverse Scale
    dummy = np.zeros((1, 6))
    dummy[0, :5] = scaled_seq[-1, :5] # Fill with last known features
    dummy[0, 5] = pred_scaled         # Put prediction in the target column

    final = scaler.inverse_transform(dummy)[0, 5]
    
    # Return clipped and rounded result
    return round(float(np.clip(final, 0, 650)), 2)

# ================= ROUTES =================

@app.post("/predict")
async def predict(data: PredictRequest):
    result = run_inference(data)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Model '{data.model_name}' not loaded or misspelled.")
    return {"predicted_load_mw": result, "status": "success"}

@app.get("/health")
def health():
    return {"loaded_models": list(models_data.keys()), "status": "healthy"}

@app.get("/")
def home():
    return {"message": "Power Forecasting API Running. Access /docs for the Swagger UI."}
