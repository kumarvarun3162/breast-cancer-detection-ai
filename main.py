from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from inference import load_model, load_config, predict

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Breast Cancer Detection API",
    description = "Upload a mammogram PNG/JPG and get cancer prediction with confidence score.",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],  # restrict to your frontend domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Load model once at startup — not on every request
model  = load_model()
config = load_config()

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status" : "running",
        "model"  : "EfficientNet-B0",
        "version": "1.0.0",
        "docs"   : "/docs"
    }


@app.get("/health")
def health():
    return {
        "status"   : "healthy",
        "model_auc": config.get("test_auc", "N/A"),
        "threshold": config.get("best_threshold", "N/A")
    }


@app.post("/predict")
async def predict_cancer(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPG images are accepted."
        )

    # Validate file size (max 20MB)
    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    try:
        result = predict(model, config, image_bytes)
        return JSONResponse(content={
            "status"  : "success",
            "filename": file.filename,
            **result
        })

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info")
def model_info():
    return {
        "architecture"  : config.get("model_arch", "efficientnet_b0"),
        "image_size"    : config.get("img_size", 224),
        "test_auc"      : config.get("test_auc", "N/A"),
        "val_auc"       : config.get("val_auc", "N/A"),
        "sensitivity"   : config.get("sensitivity", "N/A"),
        "specificity"   : config.get("specificity", "N/A"),
        "threshold"     : config.get("best_threshold", "N/A"),
        "epochs_trained": config.get("epochs_trained", "N/A"),
        "dataset"       : "VinDr-Mammo (20,000 mammograms)",
        "classes"       : ["Benign", "Malignant"]
    }


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)