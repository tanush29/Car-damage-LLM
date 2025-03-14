from fastapi import FastAPI, File, UploadFile
from src.predict import predict_and_generate_heatmap
import shutil
import os

app = FastAPI()

@app.post("/predict/")
async def predict_car_damage(file: UploadFile = File(...)):
    input_image_path = f"../images/input/{file.filename}"
    output_image_path = f"../images/output/gradcam_{file.filename}"

    # Save uploaded image
    with open(input_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model_path = "../models/car_damage_detector_model.h5"

    result = predict_and_generate_heatmap(
        input_image_path, model_path, output_image_path
    )

    return result
