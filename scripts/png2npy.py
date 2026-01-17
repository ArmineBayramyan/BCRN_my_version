# import os
# import argparse
# import skimage.io as sio
# import numpy as np
#
# parser = argparse.ArgumentParser(description='Pre-processing .png images')
# parser.add_argument('--pathFrom', default='/root/autodl-tmp/X2',
#                     help='directory of images to convert')
# parser.add_argument('--pathTo', default='/root/autodl-tmp/npy/DIV2K_train_LR_bicubic/X2',
#                     help='directory of images to save')
# parser.add_argument('--split', default=True,
#                     help='save individual images')
# parser.add_argument('--select', default='',
#                     help='select certain path')
#
# args = parser.parse_args()
# print(args)
# for (path, dirs, files) in os.walk(args.pathFrom):
#     print(path)
#     targetDir = os.path.join(args.pathTo, path[len(args.pathFrom) + 1:])
#     if len(args.select) > 0 and path.find(args.select) == -1:
#         continue
#
#     if not os.path.exists(targetDir):
#         os.mkdir(targetDir)
#
#     if len(dirs) == 0:
#         pack = {}
#         n = 0
#         for fileName in files:
#             (idx, ext) = os.path.splitext(fileName)
#             if ext == '.png':
#                 image = sio.imread(os.path.join(path, fileName))
#                 if args.split:
#                     np.save(os.path.join(targetDir, idx + '.npy'), image)
#                 n += 1
#                 if n % 100 == 0:
#                     print('Converted ' + str(n) + ' images.')


import os
import argparse
import skimage.io as sio
import numpy as np

# Base dataset folder on your PC
BASE_DATASETS_DIR = r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\datasets"

parser = argparse.ArgumentParser(description='Convert PNG LR images to NPY (CHW) format')

# LR PNGs — read from here
parser.add_argument(
    '--pathFrom',
    default=os.path.join(BASE_DATASETS_DIR, 'Flickr2K', 'Flickr2K_HR'),
    help='directory of LR PNG images'
)

# NPY destination — save here
parser.add_argument(
    '--pathTo',
    default=os.path.join(BASE_DATASETS_DIR, 'npy', 'Flickr2K_HR'),
    help='directory for output NPY files'
)

parser.add_argument('--select', default='', help='optional substring filter')

args = parser.parse_args()
print("Reading from:", args.pathFrom)
print("Saving to   :", args.pathTo)

# Create output directory
os.makedirs(args.pathTo, exist_ok=True)

# Walk tLRough all PNG files
for root, dirs, files in os.walk(args.pathFrom):
    print("Entering directory:", root)

    # Compute output directory path
    rel = root[len(args.pathFrom):].lstrip(os.sep)
    targetDir = os.path.join(args.pathTo, rel) if rel else args.pathTo
    os.makedirs(targetDir, exist_ok=True)

    # Skip directories that do not match select filter
    if args.select and (args.select not in root):
        continue

    for fname in files:
        if not fname.lower().endswith(".png"):
            continue

        src_path = os.path.join(root, fname)
        dst_path = os.path.join(targetDir, fname.replace(".png", ".npy"))

        # Load image (HWC format)
        img = sio.imread(src_path)

        # Ensure 3 color channels (some grayscale images may exist)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        # Convert HWC → CHW (PyTorch convention)
        img = img.transpose(2, 0, 1)

        # Save NPY
        np.save(dst_path, img)

        # print("Saved:", dst_path)

print("=== Conversion Finished Successfully ===")
