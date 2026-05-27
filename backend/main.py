from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ----------------------------
# Load model and tokenizer
# ----------------------------
MODEL_PATH = "bert_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# Request schema
# ----------------------------
class TweetRequest(BaseModel):
    text: str

# ----------------------------
# Prediction function
# ----------------------------
def predict_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    pred = probs[0][1].item()

    if pred > 0.7:
        label = "Disaster"
    elif pred < 0.3:
        label = "Not Disaster"
    else:
        label = "Uncertain"

    return label, round(pred, 4)

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={}
    )

@app.post("/predict")
def predict(request: TweetRequest):
    label, confidence = predict_bert(request.text)
    return {
        "tweet": request.text,
        "prediction": label,
        "confidence": confidence
    }