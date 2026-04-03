from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

app = FastAPI(title="Power Load Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 1. MODEL ARCHITECTURES =================
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
        self.moving_avg = MovingAvg(kernel_size=25, stride=1) 
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
    def forward(self, x):
        trend_init = self.moving_avg(x)
        seasonal_init = x - trend_init
        seasonal_output = self.Linear_Seasonal(seasonal_init.transpose(1, 2))
        trend_output = self.Linear_Trend(trend_init.transpose(1, 2))
        return (seasonal_output + trend_output).transpose(1, 2)

class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(out)

class InformerForecaster(nn.Module):
    def __init__(self, input_size=6, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super(InformerForecaster, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = nn.Parameter(torch.zeros(1, 24, d_model)) 
        self.encoder_input = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.encoder_input(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :]) 
        return x

class FEDformerForecaster(nn.Module):
    def __init__(self, input_size=6, d_model=64, n_modes=12):
        super(FEDformerForecaster, self).__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.w1 = nn.Parameter(torch.randn(d_model, d_model, n_modes, 2))
        self.pos_encoder = nn.Parameter(torch.zeros(1, 24, d_model))
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, x):
        B, L, C = x.shape
        x = self.enc_embedding(x) + self.pos_encoder
        x_ft = torch.fft.rfft(x, dim=1) 
        out_ft = torch.zeros(B, L//2 + 1, self.d_model, device=x.device, dtype=torch.complex64)
        weights = torch.view_as_complex(self.w1) 
        out_ft[:, :self.n_modes, :] = torch.einsum('bjc,cdj->bjd', x_ft[:, :self.n_modes, :], weights)
        x = torch.fft.irfft(out_ft, n=L, dim=1)
        return self.decoder(x[:, -1, :])

# ================= 2. LOAD MODELS & ASSETS =================
models_data = {}
startup_diagnostics = {} # 🔥 NEW: Tracks exactly why models fail

model_classes = {
    "dlinear": DLinear,
    "bilstm": BiLSTMForecaster,
    "informer": InformerForecaster,
    "fedformer": FEDformerForecaster
}

@app.on_event("startup")
def load_assets():
    for name, m_class in model_classes.items():
        try:
            pth = f"{name}_model.pth"
            pkl = f"{name}_scaler.pkl"

            if os.path.exists(pth) and os.path.exists(pkl):
                model = m_class()
                
                # 🔥 TEMPORARILY CHANGED TO strict=False for debugging
                model.load_state_dict(torch.load(pth, map_location="cpu"), strict=False)
                model.eval()

                with open(pkl, "rb") as f:
                    scaler = pickle.load(f)

                models_data[name] = {"model": model, "scaler": scaler}
                startup_diagnostics[name] = "✅ Loaded Successfully"
            else:
                startup_diagnostics[name] = f"⚠️ MISSING FILES. Looked exactly for: '{pth}' and '{pkl}'"
        except Exception as e:
            startup_diagnostics[name] = f"❌ PYTORCH ERROR: {str(e)}"

# ================= 3. INPUT SCHEMA & INFERENCE =================
class PredictRequest(BaseModel):
    temp: float
    prev_load: float
    isHoliday: int
    month: int
    hour: int
    model_name: str

def generate_history_sequence(data):
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
        seq.append([data.temp, hist_load, data.isHoliday, data.month, h, hist_load])
    return np.array(seq)

def run_inference(data: PredictRequest):
    key = data.model_name.lower()
    if key not in models_data:
        return None

    model = models_data[key]["model"]
    scaler = models_data[key]["scaler"]

    seq = generate_history_sequence(data)
    scaled_seq = scaler.transform(seq)
    input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        if key == "dlinear":
            pred_scaled = outputs[0, 0, -1].cpu().item()
        else:
            pred_scaled = outputs[0, 0].cpu().item()

    dummy = np.zeros((1, 6))
    dummy[0, :5] = scaled_seq[-1, :5] 
    dummy[0, 5] = pred_scaled         
    final = scaler.inverse_transform(dummy)[0, 5]
    
    return round(float(np.clip(final, 0, 650)), 2)

@app.post("/predict")
async def predict(data: PredictRequest):
    result = run_inference(data)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Model '{data.model_name}' not loaded.")
    return {"predicted_load_mw": result, "status": "success"}

# 🔥 NEW: Diagnostic Health Check
@app.get("/health")
def health():
    files_in_directory = os.listdir(".")
    return {
        "status": "online",
        "models_ready": list(models_data.keys()),
        "diagnostics": startup_diagnostics,
        "files_found_on_server": files_in_directory
    }

@app.get("/")
def home():
    return {"message": "API Running. Go to /health to view diagnostics."}
