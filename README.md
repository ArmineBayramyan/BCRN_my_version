# BCRN – Modified Training & Evaluation Pipeline

This repository contains a **personal modified implementation of BCRN**.

This codebase is intended for **research, experimentation, and coursework**.


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
