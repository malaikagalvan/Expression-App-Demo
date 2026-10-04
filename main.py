import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from model_predict import EmotionClassifier

app = FastAPI(title="Emotion Classifier API")

# Serve CSS/JS static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

classifier = EmotionClassifier()

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        top_emotion_1, top_emotion_2, scores = classifier.get_result(temp_path)
        
        return {
            "prediction": f"Random Forest: {top_emotion_1.title()} | Neural Network: {top_emotion_2.title()}",
            "random_forest_prediction": top_emotion_1,
            "neural_network_prediction": top_emotion_2,
            "confidence": scores
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)