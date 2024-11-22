import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rembg import remove

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Maximum file size (in bytes) - set to 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/remove-bg")
async def remove_background_api(file: UploadFile = File(...)):
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024):.2f} MB")
    
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)