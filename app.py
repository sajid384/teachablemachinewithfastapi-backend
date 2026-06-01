from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

import uvicorn
import os
import uuid
import numpy as np
import pickle

from PIL import Image

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.linear_model import LogisticRegression

# =========================================
# APP
# =========================================
app = FastAPI()

# =========================================
# PATHS
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_DIR = os.path.join(BASE_DIR, "model")

TEMP_DIR = os.path.join(BASE_DIR, "temp")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

CLASS_PATH = os.path.join(MODEL_DIR, "classes.pkl")

os.makedirs(DATASET_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(TEMP_DIR, exist_ok=True)

# =========================================
# MOBILENET V3 FEATURE EXTRACTOR
# =========================================
feature_extractor = MobileNetV3Small(

    weights="imagenet",

    include_top=False,

    pooling="avg",

    input_shape=(224, 224, 3)

)

# =========================================
# IMAGE PREPROCESS
# =========================================
def prepare_image(img_path):

    img = Image.open(img_path).convert("RGB")

    img = img.resize((224, 224))

    x = img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    return x

# =========================================
# EXTRACT FEATURES
# =========================================
def extract_features(img_path):

    x = prepare_image(img_path)

    features = feature_extractor.predict(x, verbose=0)

    return features.flatten()

# =========================================
# HOME ROUTE
# =========================================
@app.get("/")
def home():

    return {
        "message": "FastAPI Backend Running"
    }

# =========================================
# UPLOAD DATASET
# =========================================
@app.post("/upload-sample")
async def upload_sample(

    class_name: str = Form(...),

    files: list[UploadFile] = File(...)

):

    try:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        os.makedirs(class_dir, exist_ok=True)

        saved = 0

        for file in files:

            ext = file.filename.split(".")[-1]

            filename = f"{uuid.uuid4()}.{ext}"

            save_path = os.path.join(
                class_dir,
                filename
            )

            with open(save_path, "wb") as f:

                f.write(await file.read())

            saved += 1

        return {
            "message": f"{saved} images uploaded",
            "class_name": class_name
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# =========================================
# TRAIN MODEL
# =========================================
@app.post("/train")
def train_model():

    try:

        X = []

        y = []

        classes = []

        class_folders = [

            d for d in os.listdir(DATASET_DIR)

            if os.path.isdir(
                os.path.join(DATASET_DIR, d)
            )
        ]

        if len(class_folders) < 2:

            return JSONResponse(

                status_code=400,

                content={
                    "error": "Need minimum 2 classes"
                }
            )

        for idx, class_name in enumerate(class_folders):

            class_dir = os.path.join(
                DATASET_DIR,
                class_name
            )

            images = os.listdir(class_dir)

            if len(images) == 0:
                continue

            classes.append(class_name)

            for img_name in images:

                img_path = os.path.join(
                    class_dir,
                    img_name
                )

                try:

                    features = extract_features(
                        img_path
                    )

                    X.append(features)

                    y.append(idx)

                except:
                    pass

        if len(X) == 0:

            return JSONResponse(

                status_code=400,

                content={
                    "error": "No training images found"
                }
            )

        X = np.array(X)

        y = np.array(y)

        # =====================================
        # Logistic Regression
        # =====================================
        classifier = LogisticRegression(

            max_iter=1000

        )

        classifier.fit(X, y)

        # =====================================
        # SAVE MODEL
        # =====================================
        with open(MODEL_PATH, "wb") as f:

            pickle.dump(classifier, f)

        with open(CLASS_PATH, "wb") as f:

            pickle.dump(classes, f)

        return {

            "message": "Training completed successfully",

            "classes": classes,

            "training_images": len(X)

        }

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }
        )

# =========================================
# PREDICT
# =========================================
@app.post("/predict")
async def predict(

    file: UploadFile = File(...)

):

    try:

        if not os.path.exists(MODEL_PATH):

            return JSONResponse(

                status_code=400,

                content={
                    "error": "Train model first"
                }
            )

        # =====================================
        # LOAD MODEL
        # =====================================
        with open(MODEL_PATH, "rb") as f:

            classifier = pickle.load(f)

        with open(CLASS_PATH, "rb") as f:

            classes = pickle.load(f)

        # =====================================
        # SAVE TEMP IMAGE
        # =====================================
        ext = file.filename.split(".")[-1]

        filename = f"{uuid.uuid4()}.{ext}"

        temp_path = os.path.join(
            TEMP_DIR,
            filename
        )

        with open(temp_path, "wb") as f:

            f.write(await file.read())

        # =====================================
        # EXTRACT FEATURES
        # =====================================
        features = extract_features(
            temp_path
        )

        features = np.expand_dims(
            features,
            axis=0
        )

        # =====================================
        # PREDICT
        # =====================================
        prediction = classifier.predict(features)[0]

        probabilities = classifier.predict_proba(
            features
        )[0]

        confidence = round(
            float(np.max(probabilities) * 100),
            2
        )

        predicted_class = classes[prediction]

        # =====================================
        # DELETE TEMP FILE
        # =====================================
        if os.path.exists(temp_path):

            os.remove(temp_path)

        return {

            "prediction": predicted_class,

            "confidence": confidence

        }

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={
                "error": str(e)
            }
        )

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )

