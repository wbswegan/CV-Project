# Report Package Notes

## Folder Structure

The report package is organized as follows:

- `main.tex`: main LaTeX entry point
- `references.bib`: BibTeX bibliography used by the report
- `sections/`: chapter files that are included by `main.tex`
- `figures/`: copied current figures and one generated architecture diagram

The report is configured for Times New Roman through XeLaTeX.

## How to Compile

From the `report` directory, run:

```powershell
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

The expected output is `main.pdf`.

If `xelatex` reports that it cannot write `main.pdf`, close any PDF viewer that is holding the file open and run the commands again. This can happen on Windows even when the LaTeX source itself is valid.

## Main Project Materials Used

The report is based on the latest valid project state, not on outdated experiment remnants. The main evidence sources are:

- current data preparation script: `prepare_data.py`
- current training and test entry point: `main.py`
- current model definition: `src/model.py`
- current data loader and configuration: `src/data_loader.py`, `src/config.py`
- current web backend and history module: `app.py`, `src/history_store.py`
- regenerated figures in `figures/current_eda/`
- current `classification_report.txt` generated from the latest available weight file on the current grouped test set
- local smoke tests for `/health` and `/predict`

## Automatically Organized vs. Human-Review Items

The following content was generated or organized automatically from the current repository:

- the LaTeX report structure
- the copied report figures
- the architecture diagram in `report/figures/system_architecture.png`
- the front-end screenshot in `report/figures/ui_home_cropped.png`
- the wording of the report sections
- the metric table values taken from the current generated classification report

The following items should still be reviewed by a team member before final submission:

- author names and student IDs in `main.tex`
- the tutor line (`Prof. Guo`) in case the course requires a different cover format
- whether the current weight file should be treated as the final official experiment checkpoint
- any course-specific formatting rules not provided in the repository
- whether the team wants to replace the current automatically captured UI screenshot with a manually prepared one

## Dataset Cleaning and Preparation Traceability

Directly confirmed from code and current folders:

- data source is the HAM10000 image archive plus `HAM10000_metadata.csv`
- labels are taken from the `dx` field in the metadata
- the current pipeline creates `archive/Training` and `archive/Testing`
- the current split is grouped by `lesion_id` and stratified by class
- current organized data show zero image overlap and zero lesion overlap across train and test
- training uses augmentation, testing and inference use deterministic resize and normalization

Conservative statements based on available evidence:

- all metadata rows matched image files in the current organized dataset
- no explicit evidence of corrupted-image removal or manual relabeling was found
- no explicit evidence of duplicate filtering beyond lesion-level split control was found

## Algorithm Description Traceability

Directly confirmed from code:

- the model uses a ResNet18 backbone
- training initializes the backbone with pretrained weights
- an SE-style channel attention block is inserted before pooling
- the custom classifier head outputs seven classes
- training uses cross-entropy loss, Adam, and cosine annealing
- inference applies softmax and returns the top class with a confidence score
- the web backend also uses a simple blank-image heuristic and a confidence threshold for uncertain inputs

Recommended human re-check:

- whether the team wants to describe the model strictly as `SE-ResNet18` or as `ResNet18 with an SE-style attention module`
- whether the current weight file was trained fully under the present grouped split or is an earlier checkpoint reused for demonstration

## Important Caution About Results

The report uses only the newly regenerated figures in `figures/current_eda/` and does not rely on the older `figures/figures/` experiment branch. This is intentional because the older figures were tied to a different split and an outdated train/validation workflow.

The quantitative results reported in the LaTeX document are valid as evaluations of the current provided weight file on the current grouped test set. However, the repository timestamps suggest that the weight file may predate the latest refreshed grouped split. For a fully controlled final experiment, the safest next step is to retrain on the current split and regenerate all evaluation outputs.
