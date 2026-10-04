import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from rf_predict import EmotionClassifier

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
        prediction = classifier.get_result(temp_path)
        percentages = classifier.prediction_percentage(temp_path)
        return {
            "prediction": prediction,
            "confidence": percentages
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)