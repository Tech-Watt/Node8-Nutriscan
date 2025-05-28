from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

model_path = r'C:\Users\FELIX SAM(TECH WATT)\Desktop\NUTRISCAN\Tensorflow CNN model\model.h5'
model = tf.keras.models.load_model(model_path)
class_names = ['malnourish', 'nourish']

app = FastAPI()

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        image_np = preprocess_image(image_bytes)
        predictions = model.predict(image_np)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        return {'predicted_class': predicted_class, 'confidence': confidence}
    except:
        return {'error':'try with another image'}

def preprocess_image(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil = image_pil.resize((224, 224))
    img_array = np.array(image_pil) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array