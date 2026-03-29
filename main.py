from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import numpy as np
import os

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODELS =================

class DLinear(nn.Module):
    def __init__(self, seq_len=24, pred_len=1):
        super(DLinear, self).__init__()
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal = self.Linear_Seasonal(x[:, :, -1])
        trend = self.Linear_Trend(x[:, :, -1])
        return (seasonal + trend)


class BiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_ff,
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

# ================= LOAD MODELS =================

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
            pkl = f"{name}_scaler.pkl"

            if os.path.exists(pth) and os.path.exists(pkl):
                model = m_class()
                model.load_state_dict(torch.load(pth, map_location="cpu"), strict=False)
                model.eval()

                with open(pkl, "rb") as f:
                    scaler = pickle.load(f)

                models_data[name] = {"model": model, "scaler": scaler}
                print(f"✅ {name.upper()} loaded")
            else:
                print(f"⚠️ Missing files for {name}")
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

# ================= 🔥 KEY FIX: REALISTIC HISTORY =================

def generate_history_sequence(data):
    base = data.prev_load
    hour = data.hour
    seq = []

    for i in range(24):
        h = (hour - 23 + i) % 24

        # realistic daily curve
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

# ================= INFERENCE =================

def run_inference(data: PredictRequest):
    key = data.model_name.lower()
    if key not in models_data:
        return None

    model = models_data[key]["model"]
    scaler = models_data[key]["scaler"]

    # 🔥 Generate proper 24h history
    seq = generate_history_sequence(data)

    # scale
    scaled_seq = scaler.transform(seq)
    input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)

    # predict
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()[0]

    # inverse scale (put prediction in target column)
    dummy = np.zeros((1,6))
    dummy[0,:5] = scaled_seq[-1,:5]
    dummy[0,5] = pred_scaled

    final = scaler.inverse_transform(dummy)[0,5]
    return round(float(np.clip(final, 0, 650)), 2)

# ================= ROUTES =================

@app.post("/predict")
async def predict(data: PredictRequest):
    result = run_inference(data)
    if result is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    return {"predicted_load_mw": result, "status": "success"}

@app.get("/health")
def health():
    return {"loaded_models": list(models_data.keys())}

@app.get("/")
def home():
    return {"message": "Power Forecasting API Running"}
