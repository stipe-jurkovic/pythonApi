from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from returnCoordsFromImage import return_coords_from_image

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

    coords = return_coords_from_image(img)
    print(f"[INFO] Coordinates extracted: {coords}")

    return JSONResponse({
        "message": "Image processed successfully",
        "filename": image.filename,
        "coords": coords,
    })