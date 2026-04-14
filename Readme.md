# CDS540 Skin Lesion Classification Demo

This project is a local computer vision demo for classifying HAM10000 skin lesion images with an attention-enhanced ResNet18 model.

## Project Depoly (UI For shown)

<img width="1057" height="630" alt="image" src="https://github.com/user-attachments/assets/75dcbcc0-d59a-4310-9978-478be751eb94" />


## Project Structure

```text
.
|-- app.py                 # FastAPI backend and local demo page entry
|-- main.py                # Training and evaluation workflow
|-- prepare_data.py        # Dataset split and folder preparation
|-- index.html             # Demo page served by FastAPI
|-- script.js              # Frontend inference logic
|-- style.css              # Frontend styling
|-- models/
|   `-- skin_cancer_model.pth
`-- src/
    |-- config.py
    |-- data_loader.py
    |-- labels.py
    |-- model.py
    `-- utils.py
```

## Environment Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Local Demo

Start the local demo server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The FastAPI app now serves both the backend API and the frontend page, so you only need one command for the full demo.

Each prediction is also saved locally in a SQLite history store so the demo page can browse previous detections.

For Windows users, a helper launcher is also included:

```bat
start_demo.bat
```

## Training and Evaluation

Prepare the dataset folders from the original HAM10000 metadata and images:

```bash
python prepare_data.py
```

Train from scratch:

```bash
python main.py --mode train
```

Evaluate the saved weights:

```bash
python main.py --mode test
```

## Dataset Notes

- The project uses the HAM10000 metadata table and image folders in `archive/`.
- The GitHub code version does not include the full `archive/` dataset because of repository size. Place the HAM10000 metadata and image folders under `archive/` before running `prepare_data.py`.
- `prepare_data.py` now uses grouped splitting by `lesion_id`, which is safer than random image-level splitting and helps reduce train/test leakage.
- The included pretrained weight file can still be used for local inference, but if you rebuild the dataset split you should retrain before reporting final metrics.

## Known Limitations

- Class imbalance is still significant, especially for `nv` vs smaller classes like `df` and `vasc`.
- The project currently uses train/test only; there is no dedicated validation split in the training loop yet.
- This system is for coursework demonstration only and is not a medical device.
