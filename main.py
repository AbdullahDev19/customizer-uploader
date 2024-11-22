import io
import sys
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add the U-2-Net directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'U-2-Net'))

# Now we can import the U2NET model
from model.u2net import U2NET

app = FastAPI()

# Load the U-2-Net model
model_dir = './U-2-Net/saved_models/u2net/u2net.pth'

def load_model(model_path):
    model = U2NET(3, 1)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load the model")
    return model

model = load_model(model_dir)
model.eval()

def remove_background(input_image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(input_image).unsqueeze(0)

    # Predict the mask
    with torch.no_grad():
        output = model(image)
        pred = output[0].squeeze().numpy()

    # Post-process the mask
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred - mi) / (ma - mi)
    pred = np.where(pred > 0.5, 255, 0).astype(np.uint8)

    # Apply the mask to the original image
    mask = Image.fromarray(pred).resize(input_image.size, resample=Image.LANCZOS)
    output_image = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
    output_image.paste(input_image, (0, 0), mask)

    return output_image

@app.post("/remove-bg")
async def remove_background_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        output_image = remove_background(input_image)
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)