import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model.model import BluePrintConvNeXt_SR

# Testing settings

parser = argparse.ArgumentParser(description='MSR')
parser.add_argument("--test_hr_folder", type=str, default=r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\datasets\test\Urban 100\X2 Urban100\X2\HIGH X2 Urban",
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default=r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\datasets\test\Urban 100\X2 Urban100\X2\LOW X2 Urban",
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/final_best_from_200_new/Urban100/x2')
parser.add_argument("--checkpoint", type=str, default=r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\final_checkpoint_x2_new\best.pth",
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()


cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder

ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))


model = BluePrintConvNeXt_SR(upscale_factor=opt.upscale_factor).to(device)

ckpt = torch.load(opt.checkpoint, map_location=device, weights_only=False)

if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=True)
model.eval()


i = 0

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    # ----- HR image -----
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)

    # ----- LR path (FIXED) -----
    base_name = os.path.basename(imname)         
    base_root, _ = os.path.splitext(base_name)
    lr_name = base_root.replace('HR', 'LR') + ext
    lr_path = os.path.join(opt.test_lr_folder, lr_name)

    im_l = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    if im_l is None:
        print(f"[ERROR] Could not read LR image: {lr_path}")
        continue
    im_l = im_l[:, :, [2, 1, 0]]  # BGR -> RGB

    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)

    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    # out_img = utils.tensor2np(out.detach()[0])
    out = out.clamp(0, 1)
    out_img = (out[0].permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)  # RGB uint8

    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)

    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img

    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)

    output_path = os.path.join(opt.output_folder,
                               os.path.splitext(base_name)[0] + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_path, out_img[:, :, [2, 1, 0]])
    i += 1

print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(
    np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))