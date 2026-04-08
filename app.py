import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.config import (  # noqa: E402
    API_HOST,
    API_PORT,
    DEVICE,
    FRONTEND_INDEX,
    FRONTEND_SCRIPT,
    FRONTEND_STYLE,
    HISTORY_PAGE_SIZE,
    IMG_SIZE,
    MODEL_PATH,
)
from src.history_store import (  # noqa: E402
    delete_history_record,
    ensure_history_storage,
    get_history_image_path,
    get_history_record,
    list_history_records,
    save_history_record,
)
from src.labels import CLASS_NAMES, DISEASE_INFO, NUM_CLASSES  # noqa: E402
from src.model import get_model  # noqa: E402

app = FastAPI(
    title="Skin Lesion Classification Demo",
    version="1.2.0",
    description=(
        "FastAPI backend and local demo page for HAM10000 skin lesion classification, "
        "including local prediction history."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = None
model_error = None


def load_inference_model():
    global model, model_error

    if model is not None or model_error is not None:
        return

    if not os.path.exists(MODEL_PATH):
        model_error = f"Model weights not found: {MODEL_PATH}"
        return

    try:
        inference_model = get_model(NUM_CLASSES, pretrained=False).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        inference_model.load_state_dict(state_dict)
        inference_model.eval()
        model = inference_model
        print(f"Inference model loaded from {MODEL_PATH}")
    except Exception as exc:  # pragma: no cover - startup diagnostics
        model_error = f"Failed to load model weights: {exc}"
        print(model_error)


def ensure_demo_asset(path, asset_name):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{asset_name} not found.")
    return FileResponse(path)


def persist_history_entry(image, original_filename, result_payload):
    record = save_history_record(image, original_filename or "uploaded_image", result_payload)
    result_payload["history_id"] = record["id"]
    return result_payload


ensure_history_storage()
load_inference_model()


@app.get("/")
async def index():
    return ensure_demo_asset(FRONTEND_INDEX, "Frontend page")


@app.get("/script.js")
async def script():
    return ensure_demo_asset(FRONTEND_SCRIPT, "Frontend script")


@app.get("/style.css")
async def style():
    return ensure_demo_asset(FRONTEND_STYLE, "Frontend stylesheet")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model_error is None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_error,
        "device": str(DEVICE),
        "history_ready": True,
    }


@app.get("/history")
async def history(limit: int = HISTORY_PAGE_SIZE):
    clamped_limit = max(1, min(limit, 100))
    return {"items": list_history_records(limit=clamped_limit)}


@app.get("/history/{record_id}")
async def history_detail(record_id: int):
    record = get_history_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found.")
    return record


@app.get("/history/{record_id}/image")
async def history_image(record_id: int):
    image_path = get_history_image_path(record_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="History image not found.")
    return FileResponse(image_path)


@app.delete("/history/{record_id}")
async def history_delete(record_id: int):
    deleted = delete_history_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found.")
    return {"deleted": True, "id": record_id}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_inference_model()
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model is unavailable.")

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    img_np = np.array(image)
    avg_r = np.mean(img_np[:, :, 0])
    avg_g = np.mean(img_np[:, :, 1])
    avg_b = np.mean(img_np[:, :, 2])
    color_diff = abs(avg_r - avg_g) + abs(avg_g - avg_b) + abs(avg_r - avg_b)

    if avg_r > 200 and color_diff < 20:
        return persist_history_entry(
            image=image,
            original_filename=file.filename,
            result_payload={
                "prediction": "Invalid input",
                "confidence": "0%",
                "message": "The uploaded image looks like a blank document or other non-skin content.",
                "status": "invalid_input",
            },
        )

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        conf, pred = torch.max(probabilities, 1)

    confidence_value = conf.item()
    threshold = 0.5
    if confidence_value < threshold:
        return persist_history_entry(
            image=image,
            original_filename=file.filename,
            result_payload={
                "prediction": "Unrecognized",
                "confidence": f"{confidence_value * 100:.2f}%",
                "message": "The uploaded image is too uncertain for a reliable skin lesion prediction.",
                "status": "unrecognized",
            },
        )

    class_id = CLASS_NAMES[pred.item()]
    return persist_history_entry(
        image=image,
        original_filename=file.filename,
        result_payload={
            "class": DISEASE_INFO[class_id],
            "confidence": confidence_value,
            "raw_label": class_id,
            "message": "Prediction completed.",
            "status": "success",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
