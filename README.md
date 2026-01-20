# BCRN – Modified Training & Evaluation Pipeline

This repository contains a **personal modified implementation of BCRN**.

This codebase is intended for **research, experimentation, and coursework**.


---

# Clone the repository

```bash
git clone https://github.com/ArmineBayramyan/BCRN_my_version.git
cd BCRN_my_version
```

---

# Create and activate virtual environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

# Install dependencies

```bash
pip install torch torchvision numpy opencv-python scikit-image tensorboard tqdm
```

---

## Training

Train a ×2 super-resolution model:

```bash
python final_train.py
```

## Testing / Evaluation
Run evaluation on a test dataset:

```bash
python BCRN_test.py
```

## TensorBoard
Logs are saved under runs/.

```bash
tensorboard --logdir runs
```
