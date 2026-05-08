# Node8 Nutriscan

Node8 Nutriscan is a Python-based malnutrition data collection and prediction project. It includes a Streamlit app for collecting child nutrition data, a SQLite database schema for storing records, and a FastAPI endpoint for running image-based nutrition classification with a TensorFlow model.

## Project Structure

| File | Description |
| --- | --- |
| `Data_Collection_App.py` | Streamlit app for collecting nourished and malnourished child records. |
| `database_schema.py` | SQLAlchemy database models for `malnurish` and `nurish` tables. |
| `model_Api.py` | FastAPI image prediction endpoint using a TensorFlow/Keras model. |
| `malnutrition.db` | SQLite database file used by the Streamlit app. |
| `requirements.txt` | Core dependencies for running the project. |
| `all requirements` | Extended dependency list with TensorFlow, PyTorch, and plotting packages. |
| `images/` | Image assets used in the Streamlit interface. |

## Features

- Streamlit data collection form
- Separate workflows for malnourished and nourished records
- SQLite database storage
- SQLAlchemy ORM models
- Image upload fields for face, hair, hands, and legs
- FastAPI `/predict` endpoint for image classification
- TensorFlow/Keras model loading
- Simple image preprocessing with Pillow and NumPy

## Requirements

- Python 3.8 or newer
- Streamlit
- SQLAlchemy
- FastAPI
- Uvicorn
- Pillow
- TensorFlow, if you want to run the prediction API

Install the core dependencies:

```bash
pip install -r requirements.txt
```

For the full ML stack, install the packages listed in `all requirements`:

```bash
pip install streamlit sqlalchemy uvicorn psycopg2-binary torch tensorflow matplotlib torchvision fastapi python-multipart pillow
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Tech-Watt/Node8-Nutriscan.git
cd Node8-Nutriscan
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

Start the data collection app:

```bash
streamlit run Data_Collection_App.py
```

The app lets you choose between:

- `Malnourish`
- `Nourished`

Each form collects:

- Age
- Weight
- Height
- Mid-lower hand circumference
- Skin type
- Hair type
- Eye type
- Oedema
- Angular stomatitis
- Cheilosis
- Bowlegs
- Location
- Image uploads

Submitted records are saved into `malnutrition.db`.

## Database

The SQLite database is configured in `Data_Collection_App.py`:

```python
SQLALCHEMY_DATABASE_URL = 'sqlite:///./malnutrition.db'
```

The database schema includes two tables:

- `malnurish`
- `nurish`

The models are defined in `database_schema.py`.

## Run the Prediction API

`model_Api.py` exposes a FastAPI endpoint:

```text
POST /predict
```

Start the API with:

```bash
uvicorn model_Api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Use the Swagger UI to upload an image and test prediction.

## Model Path Setup

The API currently loads a local model from this path:

```python
model_path = r'C:\Users\FELIX SAM(TECH WATT)\Desktop\NUTRISCAN\Tensorflow CNN model\model.h5'
```

Before running the API, update `model_path` in `model_Api.py` so it points to your own `.h5` model file:

```python
model_path = r'path\to\your\model.h5'
```

The model predicts one of these classes:

```python
class_names = ['malnourish', 'nourish']
```

## Notes

- Keep medical and child health data private and secure.
- Do not commit sensitive records, API keys, or private datasets.
- The FastAPI model file is not included, so update the model path before running predictions.
- The project uses local SQLite for development; consider PostgreSQL or another managed database for production.
- This project is for learning and prototyping, not clinical diagnosis.

## Author

Created by [Tech Watt](https://github.com/Tech-Watt).
