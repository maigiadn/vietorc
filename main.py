import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

app = FastAPI(title="VietOCR API")

# Load configuration
config = Cfg.load_config_from_name('vgg_transformer')

# Hardware Detection: Force CPU
config['device'] = 'cpu'

# Initialize Predictor
try:
    detector = Predictor(config)
except Exception as e:
    print(f"Warning: Model initialization failed. Details: {e}")
    detector = None

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Read image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Convert image to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if detector is None:
            return {"status": "error", "message": "Predictor model is not initialized."}
            
        # Predict
        extracted_text = detector.predict(image)
        
        return {
            "status": "success",
            "extracted_text": str(extracted_text)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
