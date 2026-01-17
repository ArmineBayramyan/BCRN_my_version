import os
import shutil
import numpy as np
import skimage.io as sio

# ---- CONFIG ----
BASE_DATASETS_DIR = r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\datasets"
NPY_ROOT = os.path.join(BASE_DATASETS_DIR, "npy")


def convert_folder(src_dir, dst_dir):
    """
    Convert all PNG images in src_dir to NPY (HWC) in dst_dir.
    Keeps subfolder structure.
    """
    print(f"\nConverting:\n  from {src_dir}\n  to   {dst_dir}")
    for root, _, files in os.walk(src_dir):
        rel = root[len(src_dir):].lstrip(os.sep)
        out_dir = os.path.join(dst_dir, rel) if rel else dst_dir
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(".png"):
                continue

            src_path = os.path.join(root, fname)
            dst_path = os.path.join(out_dir, fname.replace(".png", ".npy"))

            img = sio.imread(src_path)  # HWC

            # If grayscale, make it 3-channel
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            np.save(dst_path, img)  # save HWC
            print("  Saved:", dst_path)

    print("Done:", dst_dir)


def main():
    # 1) Delete existing NPY root
    if os.path.exists(NPY_ROOT):
        print("Removing existing NPY root:", NPY_ROOT)
        shutil.rmtree(NPY_ROOT)
    os.makedirs(NPY_ROOT, exist_ok=True)
    print("Created fresh NPY root:", NPY_ROOT)

    # 2) Define all source/destination pairs
    pairs = [
        # DIV2K HR
        (
            os.path.join(BASE_DATASETS_DIR, "DIV2K", "DIV2K_train_HR"),
            os.path.join(NPY_ROOT, "DIV2K_train_HR"),
        ),
        # DIV2K LR x2  (bicubic)
        (
            os.path.join(BASE_DATASETS_DIR, "DIV2K", "DIV2K_train_LR_bicubic"),
            os.path.join(NPY_ROOT, "DIV2K_train_LR_bicubic", "X2"),
        ),
        # Flickr2K HR
        # (
        #     os.path.join(BASE_DATASETS_DIR, "Flickr2K_HR"),
        #     os.path.join(NPY_ROOT, "Flickr2K_HR"),
        # ),
        # # Flickr2K LR x2
        # (
        #     os.path.join(BASE_DATASETS_DIR, "Flickr2K_LR_x2"),
        #     os.path.join(NPY_ROOT, "Flickr2K_LR_x2"),
        # ),
    ]

    # 3) Convert each dataset
    for src, dst in pairs:
        if not os.path.isdir(src):
            print(f"\n[WARNING] Source folder not found, skipping:\n  {src}")
            continue
        convert_folder(src, dst)

    print("\n=== ALL CONVERSIONS FINISHED ===")


if __name__ == "__main__":
    main()
