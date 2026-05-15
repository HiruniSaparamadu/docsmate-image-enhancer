# Docsmate Image Enhancer
An AI-powered image enhancement API built with FastAPI, OpenCV, and PyTorch.

## Features
- Enhance blurry and low-quality images
- AI upscaling using RealESRGAN
- Face restoration using GFPGAN
- PDF enhancement support
- Denoising and sharpening
- Automatic document/photo detection

## Technologies Used
- Python
- FastAPI
- OpenCV
- PyTorch
- PyMuPDF

## Environment Variables

Create a `.env` file in the project root:

```env
ENHANCE_SCALE=2
ENABLE_FACE=true
ENHANCE_TIMEOUT=120
```

### Variable Explanation

| Variable | Description |

| ENHANCE_SCALE | Upscaling factor (2 = faster, 4 = higher quality) |
| ENABLE_FACE | Enables GFPGAN face restoration |
| ENHANCE_TIMEOUT | Maximum processing time in seconds |


## Project Setup
### 1. Clone the repository

```bash
git clone https://github.com/HiruniSaparamadu/docsmate-image-enhancer.git
```

### 2. Open the project folder

```bash
cd docsmate-image-enhancer
```

### 3. Create virtual environment

```bash
python -m venv venv
```

### 4. Activate virtual environment
#### Windows

```bash
venv\Scripts\activate
```
#### Mac/Linux

```bash
source venv/bin/activate
```

### 5. Install requirements

```bash
pip install -r requirements.txt
```

### 6. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Open the application

open index.html


## AI Models Used

- RealESRGAN
- GFPGAN
- EDSR

## Author

Hiruni Saparamadu