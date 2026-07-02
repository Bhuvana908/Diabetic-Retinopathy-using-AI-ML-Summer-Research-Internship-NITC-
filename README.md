# Diabetic Retinopathy Detection using AI/ML
### Summer Research Internship — NIT Calicut (NITC)

This repository documents an iterative research project on automated **Diabetic
Retinopathy (DR) detection and lesion analysis** from retinal fundus images. It
contains a sequence of 19 Google Colab notebooks (`DR1` → `DR19`), each one a
successive experiment that builds on the previous one — starting from a plain
CNN classifier and evolving into a **superpixel-based Graph Attention Network
(GAT)** pipeline with lesion-level interpretability.

Because every notebook is a self-contained checkpoint in this progression,
this single README explains the project as a whole: the overall goal, how the
approach evolved from DR1 to DR19, the shared pipeline/architecture, the data
and setup required to run any notebook, and how to navigate the repo.

---

## 1. Project Goal

Diabetic Retinopathy is a leading cause of preventable blindness in diabetic
patients. Grading its severity from a fundus (retina) photograph normally
requires an ophthalmologist. This project explores whether **deep learning +
graph-based lesion modeling** can:

1. Classify DR severity (typically the standard 5-class ICDR scale: No DR,
   Mild, Moderate, Severe, Proliferative DR), and
2. Localize and characterize the underlying **lesions** (microaneurysms,
   hemorrhages, exudates, etc.) driving that classification, so predictions
   are clinically interpretable rather than a black box.

## 2. How the Approach Evolved (DR1 → DR19)

| Stage | Notebooks | Approach |
|---|---|---|
| **CNN baseline** | `DR1`, `DR2` | EfficientNet-B3 image classifier on the APTOS 2019 dataset, with Albumentations augmentation, class-imbalance handling (`WeightedRandomSampler`), and **Grad-CAM** for visual explainability. SLIC superpixel segmentation introduced for later lesion analysis. |
| **Superpixel graph construction** | `DR3`, `DR4` | Fundus images are segmented into superpixels (SLIC), each superpixel becomes a graph node (color/texture features via `rgb2lab`, k-NN adjacency), and a **Graph Attention Network (`LesionGAT_v2`)** built with PyTorch Geometric is trained on top of this graph representation. |
| **Refined lesion-graph model** | `DR5` – `DR8` | `LesionGAT` architecture refined; richer node features (LBP, GLCM texture, Otsu thresholding); **focal loss** added to handle class imbalance; hybrid `SuperpixelCNN` / `DGCN` / `GCN` variants explored (`DR6`). |
| **Multi-dataset validation** | `DR9` – `DR11` | **IDRiD** dataset added alongside APTOS for cross-dataset evaluation; Google Drive integration for larger datasets; K-fold cross-validation; segmentation-style metrics (Precision, Recall, F1, IoU, Dice, Specificity) added for lesion-level evaluation. |
| **Contrastive + adaptive graphs** | `DR12` – `DR19` | Introduces `DynamicKNN` (adaptive graph edge construction) and `GCLProjector` (graph contrastive learning projector) to improve node/graph embeddings before GAT classification. Later notebooks (`DR14`–`DR19`) add more robust K-fold training, threshold optimization for best F1, and improved visualization/reporting (confusion matrices, per-class metric summaries, GradCAM-style overlays). `DR19` is the most complete and refined version of the pipeline. |

**In short:** `DR1`/`DR2` = CNN baseline → `DR3`–`DR11` = superpixel Graph
Attention Network development → `DR12`–`DR19` = contrastive learning +
adaptive graph refinement, culminating in `DR19`.

> If you only want to run **one** notebook to see the final pipeline,
> start with **`DR19/DR19.ipynb`**. For the simplest CNN baseline, use
> **`DR2/DR2.ipynb`**.

## 3. Repository Structure

```
Diabetic-Retinopathy-using-AI-ML-Summer-Research-Internship-NITC-/
├── DR1/DR1.ipynb    # EfficientNet-B3 CNN baseline + Grad-CAM
├── DR2/DR2.ipynb    # CNN baseline, refined augmentation/sampling
├── DR3/DR3.ipynb    # First superpixel graph + GAT (LesionGAT_v2)
├── DR4/DR4.ipynb    # Graph pipeline refinement
├── DR5/DR5.ipynb    # LesionGAT architecture
├── DR6/DR6.ipynb    # SuperpixelCNN / GCN / DGCN variants
├── DR7/DR7.ipynb    # LesionGAT + focal loss, texture features
├── DR8/DR8.ipynb    # LesionGAT, LBP/GLCM features, Otsu thresholding
├── DR9/DR9.ipynb    # + IDRiD dataset, segmentation metrics (F1/IoU/Dice)
├── DR10/DR10.ipynb  # Google Drive integration, class balancing
├── DR11/DR11.ipynb  # K-fold cross-validation
├── DR12/DR12.ipynb  # + DynamicKNN, GCLProjector (contrastive learning)
├── DR13/DR13.ipynb  # Refined contrastive graph pipeline
├── DR14/DR14.ipynb  # K-fold + contrastive graph, improved reporting
├── DR15/DR15.ipynb  # Further refinement
├── DR16/DR16.ipynb  # Further refinement
├── DR17/DR17.ipynb  # Further refinement
├── DR18/DR18.ipynb  # Further refinement
└── DR19/DR19.ipynb  # Most complete / final pipeline
```

Each `DRn` folder contains a single, self-contained Colab notebook — there
are no shared `.py` modules; every notebook re-defines the classes/functions
it needs.

## 4. Datasets Used

- **[APTOS 2019 Blindness Detection](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)** (Kaggle) — retinal images labeled 0–4 for DR severity. Used from `DR1` onward.
- **IDRiD Segmentation Dataset** — added from `DR9` onward for lesion-level
  segmentation masks and cross-dataset validation. Uploaded to Google Drive:
  **[IDRiD Segmentation](https://drive.google.com/file/d/1eXhbsN2jC7g7ueBbnB8-VoQUSVFX1Kpz/view?usp=sharing)**

Datasets are **not included** in this repository (they are large and
license-restricted). Each notebook downloads them at runtime via:
- the **Kaggle API** (`kaggle.json` credentials) for APTOS, and/or
- a **Google Drive** mount for IDRiD in later notebooks.

## 5. Core Techniques & Libraries

| Category | Tools |
|---|---|
| Deep learning | PyTorch, `torchvision` (EfficientNet-B3), PyTorch Geometric (`GATConv`, graph data objects) |
| Image processing | OpenCV, `scikit-image` (SLIC superpixels, LBP, GLCM texture, Otsu thresholding, `rgb2lab`) |
| Augmentation | Albumentations |
| Classical ML utilities | scikit-learn (train/test split, `KFold`, `StandardScaler`, `NearestNeighbors`, metrics) |
| Explainability | `pytorch-grad-cam` (Grad-CAM heatmaps, early notebooks) |
| Data/plots | pandas, NumPy, Matplotlib, Seaborn |
| Environment | Google Colab, Kaggle API, Google Drive |

**Modeling approach (later notebooks, DR3–DR19):**
1. Preprocess fundus image → segment into **SLIC superpixels**.
2. Extract per-superpixel features (color in Lab space, texture via LBP/GLCM).
3. Build a **graph** where superpixels are nodes and edges come from spatial/
   feature nearest-neighbors (static k-NN in early notebooks, `DynamicKNN` in
   later ones).
4. (From `DR12` on) Pass node embeddings through a **graph contrastive
   learning projector (`GCLProjector`)** to improve representation quality.
5. Classify with a **Graph Attention Network (`GATConv` layers)**, using
   focal loss / class-weighting to handle DR class imbalance.
6. Evaluate with accuracy, confusion matrix, and per-class
   Precision/Recall/F1/IoU/Dice/Specificity; visualize lesion attention.

## 6. How to Run a Notebook

All notebooks are built for **Google Colab** (each starts with an "Open in
Colab" badge) and assume a Colab/Linux environment with a GPU runtime.

1. Open the notebook of interest in Google Colab (or Jupyter with a GPU).
2. Set runtime type to **GPU**.
3. Provide credentials as required by that notebook:
   - **Kaggle API**: upload your `kaggle.json` (Kaggle account → *Create New
     API Token*) when prompted, or place it at `~/.kaggle/kaggle.json`.
   - **Google Drive** (DR9+): first upload the **IDRiD segmentation
     dataset** to your own Google Drive (see the link in §4), then run the
     `google.colab.drive.mount()` cell and authorize access so the notebook
     can read it; update the dataset zip path to match where you placed it
     in your Drive.
4. Run all cells top to bottom. Notebooks handle dataset download/extraction,
   preprocessing, training, and evaluation end-to-end.

## 7. Notes & Limitations

- Notebooks are experimental research checkpoints, not a packaged library —
  expect duplicated code across notebooks rather than shared modules.
- No pretrained model weights or dataset files are included; both must be
  obtained separately (see §4).
- No automated test suite; correctness was validated interactively within
  each Colab session.

## 8. Author / Context

Developed as part of a Summer Research Internship at the **National
Institute of Technology, Calicut (NITC)**, exploring AI/ML methods —
CNNs, superpixel segmentation, and Graph Attention Networks — for
interpretable Diabetic Retinopathy detection.
