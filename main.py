from fastapi import FastAPI, UploadFile, File
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import io

app = FastAPI()

# 1. ÉP LOAD CONFIG CHUẨN VÀ CHẠY BẰNG CPU (Cho VPS thường)
config = Cfg.load_config_from_name('vgg_seq2seq') # Dùng seq2seq cho nhẹ, vgg_transformer thì nặng hơn
config['device'] = 'cpu' # Chốt hạ chạy CPU để tránh lỗi suy luận rác
# config['cnn']['pretrained'] = False # Mở comment dòng này nếu bị lỗi load pretrain

# Khởi tạo model
predictor = Predictor(config)

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Đọc luồng nhị phân
        contents = await file.read()
        
        # 2. KHÚC CHÍ MẠNG: Ép convert về RGB để chống mù màu
        image = Image.open(io.BytesIO(contents)).convert('RGB') 
        
        # Bắt đầu đọc chữ
        text = predictor.predict(image)
        
        return {"status": "success", "extracted_text": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}
