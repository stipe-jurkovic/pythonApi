from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from returnCoordsFromImage import return_coords_from_image
import base64

app = FastAPI()

@app.post("/upload/")
async def upload_image(image: UploadFile = File(...)):
    # Vrati pokazivač na početak
    image.file.seek(0)

    # Pročitaj bytes iz uploadanog file-like objekta
    file_bytes = await image.read()
    
    # Pretvori bytes u numpy array tipa uint8
    np_arr = np.frombuffer(file_bytes, np.uint8)
    
    # Dekodiraj numpy array u OpenCV sliku
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Ne mogu dekodirati sliku"}, status_code=400)

    coords, fully_annotated_image = return_coords_from_image(img)
    print(f"[INFO] Coordinates extracted: {coords}")
    # Convert NumPy array to raw bytes
    # Encode as JPEG with compression (quality 70 out of 100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    success, jpeg_bytes = cv2.imencode(".jpg", fully_annotated_image, encode_param)
    if not success:
        return JSONResponse({"error": "Failed to encode image as JPEG"}, status_code=500)

    # Base64 encode the compressed JPEG
    img_base64 = base64.b64encode(jpeg_bytes.tobytes()).decode("utf-8")
    return JSONResponse({
        "message": "Image processed successfully",
        "filename": image.filename,
        "coords": coords,
        "fully_annotated_image": img_base64
    })