from pathlib import Path
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow import keras
from time import time
import json
import io
from PIL import Image

app = FastAPI()
path = Path(__file__).parent.resolve()

effNetv2Model = keras.models.load_model(path / "EfficientNetV2B0.h5")
effNetv1Model = keras.models.load_model(path / "EfficientNetV1B0.h5")
# vitModel = keras.models.load_model(path / "ViTTL8classification.h5") TODO:

with open(path / "classes.json", "r") as f:
    class_names = list(json.load(f).keys())

model_data = {
    'efficientnetv2' : {
        'model' : effNetv2Model,
        'preprocess' : None
    },
    'efficientnetv1' : {
        'model' : effNetv1Model,
        'preprocess' : tf.keras.applications.efficientnet.preprocess_input
    },
    'vit' : {
        'model' : vitModel,
        'preprocess' : None
    }
}

async def classify(img, model, preprocess = None):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = preprocess(img_array) if (preprocess and callable(preprocess)) else img_array

    predictions = model.predict(img_array)

    predictions_decoded = {class_names[i] : value * 100 for i, value in sorted(enumerate(predictions[0]), key=lambda val: val[1], reverse=True)}

    return predictions_decoded


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "efficientnet"):
    data = model_data[model]
    model, preprocess = data['model'], data['preprocess']

    img = Image.open(io.BytesIO(await file.read()))
    predictions = await classify(img, model, preprocess)
    return predictions

@app.post("/feedback")
async def predict(correctClass: str, file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.save(path / "data" / correctClass / str(time()) + ".jpg")
    except:
        return {"error": "Could not save image", "success": False}

    return {"success": True}