## Create a new directory for the project
mkdir -p background_remover
cd background_remover

## Create a virtual environment
python3 -m venv venv

## Activate the virtual environment
source venv/bin/activate

## Upgrade pip
pip install --upgrade pip

## Install required packages
pip install fastapi uvicorn pillow python-multipart rembg onnxruntime

### python main.py

```
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/remove-bg --output output.png
```


