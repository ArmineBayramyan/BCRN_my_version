import argparse
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import skimage.color as sc

from dataset_utils import DIV2K, Set5_val
import utils
from model.model import BluePrintConvNeXt_SR


# ----------------------------- Utils -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    ckpt_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_psnr: float,
    is_best: bool,
):
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_psnr": best_psnr,
    }
    path = os.path.join(ckpt_dir, f"epoch_{epoch}.pth")
    torch.save(state, path)
    print(f"===> Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, best_path)
        print(f"===> Updated best checkpoint: {best_path} (best_psnr={best_psnr:.4f})")


def try_resume(resume_path: str, device: torch.device, model, optimizer, scheduler):
    if not resume_path:
        return 1, -float("inf")
    if not os.path.exists(resume_path):
        print(f"WARNING: resume path not found: {resume_path} (starting fresh)")
        return 1, -float("inf")

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", -float("inf")))
        print(f"===> Resumed from {resume_path} | start_epoch={start_epoch} | best_psnr={best_psnr:.4f}")
        return start_epoch, best_psnr

    model.load_state_dict(ckpt)
    print(f"===> Loaded model weights only (old checkpoint format): {resume_path}")
    return 1, -float("inf")


# ----------------------------- Train / Valid -----------------------------
def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    grad_clip: float = 1.0,
):
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    if num_batches == 0:
        print("ERROR: training_loader has 0 batches. Dataset path / loading is wrong.")
        return float("nan"), global_step

    for it, (lr_tensor, hr_tensor) in enumerate(loader, 1):
        lr_tensor = lr_tensor.to(device, non_blocking=True)
        hr_tensor = hr_tensor.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        sr_tensor = model(lr_tensor)
        loss = criterion(sr_tensor, hr_tensor)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[NaN/Inf] epoch {epoch} iter {it}: loss invalid. Stopping this epoch.")
            return float("nan"), global_step

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        writer.add_scalar("Loss/train", loss.item(), global_step)

        if it % 100 == 0:
            print(f"===> Epoch[{epoch}]({it}/{num_batches}) Loss_L1: {loss.item():.6f}")

        # flush sometimes so TB updates live
        if it % 200 == 0:
            writer.flush()

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    epoch: int,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    scale: int,
    eval_on_y: bool,
    global_step: int,
):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    if len(loader) == 0:
        print("ERROR: testing_loader has 0 batches.")
        return float("nan"), float("nan"), float("nan")

    for lr_tensor, hr_tensor in loader:
        lr_tensor = lr_tensor.to(device, non_blocking=True)
        hr_tensor = hr_tensor.to(device, non_blocking=True)

        pre = model(lr_tensor)
        loss = criterion(pre, hr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])

        cropped_sr_img = utils.shave(sr_img, scale)
        cropped_gt_img = utils.shave(gt_img, scale)

        if eval_on_y:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img

        psnr = utils.compute_psnr(im_pre, im_label)
        ssim = utils.compute_ssim(im_pre, im_label)

        total_loss += loss.item()
        total_psnr += psnr
        total_ssim += ssim
        n += 1

    val_loss = total_loss / max(n, 1)
    val_psnr = total_psnr / max(n, 1)
    val_ssim = total_ssim / max(n, 1)

    print(f"===> Valid Epoch[{epoch}]  L1: {val_loss:.6f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")

    # Use SAME global_step axis as training (so plots align naturally)
    writer.add_scalar("Loss/val", val_loss, global_step)
    writer.add_scalar("PSNR/val", val_psnr, global_step)
    writer.add_scalar("SSIM/val", val_ssim, global_step)

    writer.flush()
    return val_loss, val_psnr, val_ssim


# ----------------------------- Main -----------------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True

    BASE_ROOT = r"C:\Users\Admin\PycharmProjects\BCRN\BCRN\datasets\npy"

    parser = argparse.ArgumentParser(description="BCRN / MSR Training")

    parser.add_argument("--root", type=str, default=BASE_ROOT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--testBatchSize", type=int, default=1)

    parser.add_argument("--nEpochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--scale", type=int, default=2, choices=[2, 4, 8])
    parser.add_argument("--patch_size", type=int, default=48)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--isY", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--pretrained", type=str, default="")

    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--resume_path", type=str, default="")

    # compatibility args
    parser.add_argument("--n_train", type=int, default=3450)
    parser.add_argument("--n_val", type=int, default=1)
    parser.add_argument("--test_every", type=int, default=1000)
    parser.add_argument("--rgb_range", type=int, default=1)
    parser.add_argument("--n_colors", type=int, default=3)
    parser.add_argument("--ext", type=str, default=".npy")
    parser.add_argument("--phase", type=str, default="train")

    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()
    print(args)

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    set_seed(args.seed)

    # -------- TensorBoard: unique run folder (prevents confusion) --------
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = args.run_name.strip() or f"final_BCRN_x{args.scale}"
    run_name = f"{base_name}_{ts}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    # force event file creation immediately:
    writer.add_text("run/info", f"run_name={run_name}", 0)
    writer.flush()

    print("TensorBoard log dir:", os.path.abspath(log_dir))
    print("Run TensorBoard with:")
    print(f'  tensorboard --logdir "{os.path.abspath("runs")}"')

    # -------- datasets --------
    print("===> Loading datasets")
    args.is_train = True
    trainset = DIV2K.div2k(args)
    testset = Set5_val.DatasetFromFolderVal(
        "Test_Datasets/Set5/",
        f"Test_Datasets/Set5_LR/x{args.scale}/",
        args.scale,
    )

    print("Trainset length:", len(trainset), "| Testset length:", len(testset))

    training_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=use_cuda,
        drop_last=True,
    )
    testing_loader = DataLoader(
        testset,
        batch_size=args.testBatchSize,
        shuffle=False,
        num_workers=args.threads,
        pin_memory=use_cuda,
    )

    print("training_loader batches:", len(training_loader))
    print("testing_loader batches:", len(testing_loader))

    # -------- model/loss --------
    model = BluePrintConvNeXt_SR(upscale_factor=args.scale).to(device)
    criterion = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # optional pretrained weights
    if args.pretrained and os.path.exists(args.pretrained):
        sd = torch.load(args.pretrained, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            model.load_state_dict(sd["model"])
        else:
            model.load_state_dict(sd)
        print("===> Loaded pretrained weights from:", args.pretrained)

    # resume
    start_epoch, best_psnr = try_resume(args.resume_path, device, model, optimizer, scheduler)

    ckpt_dir = args.ckpt_dir.strip() or f"final_checkpoint_x{args.scale}_new"
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = (start_epoch - 1) * max(len(training_loader), 1)

    # training loop
    print("===> Training")
    for epoch in range(start_epoch, args.nEpochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, global_step)

        print(f"\nEpoch {epoch}/{args.nEpochs} | LR={current_lr:.6e} | global_step={global_step}")

        train_loss, global_step = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=training_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            global_step=global_step,
            grad_clip=args.grad_clip,
        )

        val_loss, val_psnr, val_ssim = validate(
            epoch=epoch,
            model=model,
            loader=testing_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            scale=args.scale,
            eval_on_y=args.isY,
            global_step=global_step,
        )

        scheduler.step()

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint(
            ckpt_dir=ckpt_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_psnr=best_psnr,
            is_best=is_best,
        )

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()