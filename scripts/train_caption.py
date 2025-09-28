import os

from PIL import Image
from typing import List
from tqdm import tqdm

import torch
from transformers import get_cosine_schedule_with_warmup

# ========= 数据集 =========
from coco.data.coco_caption_dataset import COCOCaptionDataset, get_dataloader
from coco.model.caption_model import COCOCaptionModel


# ========= 全局配置 =========
COCO_ROOT = "C:\\Users\\61556\\Downloads\\data\\coco"  # 必须是包含 annotations/train2017/val2017 的根目录
SAVE_DIR = "./save_caption"
EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 128
NUM_WORKERS = 4  # Windows 上建议先用 0，稳定且易排障
PRINT_EVERY = 50  # 训练中每多少 step 打印一次 loss
os.makedirs(SAVE_DIR, exist_ok=True)


# ========= 工具函数 =========
def check_sentencepiece():
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        raise ImportError(
            "未检测到 sentencepiece 库。请先安装：\n"
            "  pip install sentencepiece\n"
            "安装后重启解释器/IDE 再运行。"
        )


def assert_coco_layout(root: str) -> None:
    """检查 COCO 根目录关键文件是否存在；不存在则抛出明确错误。"""
    need = [
        os.path.join(root, "annotations", "captions_train2017.json"),
        os.path.join(root, "annotations", "captions_val2017.json"),
        os.path.join(root, "train2017"),
        os.path.join(root, "val2017"),
    ]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        msg = "以下必需路径不存在，请检查 COCO_ROOT 是否正确：\n" + "\n".join(missing)
        raise FileNotFoundError(msg)


def make_image_grid(images: List[Image.Image], cols: int = 3) -> Image.Image:
    """把若干 PIL 图拼成网格。不写文字，文字单独保存到 txt。"""
    if len(images) == 0:
        raise ValueError("images 为空，无法生成网格。")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(im.resize((w, h)), (c * w, r * h))
    return grid


def build_optimizer(model, lr=1e-4, weight_decay=0.2):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "bias" in name or "LayerNorm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98), eps=1e-8
    )


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(tqdm(loader)):
        pixel_values, input_ids, labels = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(pixel_values, input_ids, labels)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1
        if step % PRINT_EVERY == 0:
            print(f"[train step {step}] loss={loss.item():.4f}")

    if steps == 0:
        print("训练 DataLoader 未产生任何 batch（可能是 dataset 为空或 batch_size 太大且 drop_last=True）。")
        return float("inf")
    return total_loss / steps


# ========= 主流程 =========
def main():
    # 依赖 & 目录自检
    check_sentencepiece()
    assert_coco_layout(COCO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据集
    train_ds = COCOCaptionDataset(COCO_ROOT, split="val", max_length=MAX_LENGTH)
    print(f"Train size: {len(train_ds)}")

    # DataLoader
    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Train batches: {len(train_loader)}")

    # 模型 & 优化器 & 调度器
    model = COCOCaptionModel().to(device)
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.2)
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(1000, total_steps // 10),
                                                num_training_steps=total_steps)

    # 创建日志文件
    log_file = os.path.join(SAVE_DIR, "train_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch}  Train loss={train_loss:.4f}")

        # 追加日志
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f}\n")

        # 保存 checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "logit_scale": float(model.logit_scale.exp().item()),
        }, ckpt_path)
        print(f"[INFO] Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 把异常也写入 save 方便排障
        err_path = os.path.join(SAVE_DIR, "run_error.log")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        raise
