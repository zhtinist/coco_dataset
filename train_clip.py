# train_coco_siglip.py
import os
import math
import json
from PIL import Image
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    SiglipProcessor,
    SiglipVisionModel,
    SiglipTextModel,
    get_cosine_schedule_with_warmup,
)

# ========= 全局配置 =========
COCO_ROOT = r"D:\datasets\coco"  # 必须是包含 annotations/train2017/val2017 的根目录
SAVE_DIR = "./save"
EPOCHS = 5
BATCH_SIZE = 32
MAX_LENGTH = 32
NUM_WORKERS = 0  # Windows 上建议先用 0，稳定且易排障
DEBUG_CHECK = True        # 是否做数据自检
DEBUG_NUM_SAMPLES = 6     # 导出前 N 个样本做可视化/文本检查
PRINT_EVERY = 50          # 训练中每多少 step 打印一次 loss
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


# ========= 数据集 =========
class CocoCaptionDataset(Dataset):
    """COCO captions：每条样本=一张图 + 一条 caption。"""
    def __init__(self, root: str, split: str, processor: SiglipProcessor, max_length: int = 32):
        self.root = root
        self.split = split
        self.processor = processor
        self.max_length = max_length

        anno_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # image_id -> filename
        self.image_files = {img["id"]: img["file_name"] for img in data["images"]}
        # (image_id, caption) 列表
        self.samples: List[Tuple[int, str]] = [(ann["image_id"], ann["caption"]) for ann in data["annotations"]]

    def __len__(self):
        return len(self.samples)

    def _resolve_path_caption(self, idx: int) -> Tuple[str, str]:
        image_id, caption = self.samples[idx]
        file_name = self.image_files[image_id]
        img_path = os.path.join(self.root, f"{self.split}2017", file_name)
        return img_path, caption

    def __getitem__(self, idx):
        img_path, caption = self._resolve_path_caption(idx)
        image = Image.open(img_path).convert("RGB")

        enc = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # 去掉 batch 维
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),   # [3,224,224]
            "input_ids": enc["input_ids"].squeeze(0),         # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ========= 模型 & 训练 =========
class SiglipDualEncoder(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.vision = SiglipVisionModel.from_pretrained(model_name)
        self.text = SiglipTextModel.from_pretrained(model_name)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # 可学习温度

    def forward(self, pixel_values, input_ids, attention_mask):
        v = self.vision(pixel_values=pixel_values).pooler_output
        t = self.text(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        v = nn.functional.normalize(v, p=2, dim=-1)
        t = nn.functional.normalize(t, p=2, dim=-1)
        logits = v @ t.T  # 余弦相似（已 L2）
        return logits * self.logit_scale.exp()


def symmetric_ce_loss(logits):
    n = logits.size(0)
    targets = torch.arange(n, device=logits.device)
    it2txt = nn.functional.cross_entropy(logits, targets)
    txt2it = nn.functional.cross_entropy(logits.T, targets)
    return (it2txt + txt2it) / 2


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
    for step, batch in enumerate(loader, 1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = symmetric_ce_loss(logits)

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
        print("⚠️ 训练 DataLoader 未产生任何 batch（可能是 dataset 为空或 batch_size 太大且 drop_last=True）。")
        return float("inf")
    return total_loss / steps


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = symmetric_ce_loss(logits)
        total_loss += loss.item()
        steps += 1
    if steps == 0:
        print("⚠️ 验证 DataLoader 未产生任何 batch。")
        return float("inf")
    return total_loss / steps


def debug_inspect_dataset(ds: CocoCaptionDataset, split: str):
    """导出前 N 个样本（图像网格 + 文本），帮助确认是否正确读到数据。"""
    n = min(DEBUG_NUM_SAMPLES, len(ds))
    print(f"[DEBUG] {split} 集样本数 = {len(ds)}，将导出前 {n} 个样本到 {SAVE_DIR}")
    if n == 0:
        with open(os.path.join(SAVE_DIR, f"debug_{split}_empty.txt"), "w", encoding="utf-8") as f:
            f.write("数据集为空，请检查 COCO_ROOT 与 annotations。\n")
        return

    images, captions = [], []
    for i in range(n):
        img_path, caption = ds._resolve_path_caption(i)
        captions.append(f"{i+1}. {os.path.basename(img_path)} :: {caption}")
        try:
            im = Image.open(img_path).convert("RGB")
            images.append(im)
        except Exception as e:
            captions.append(f"   (打开图像失败: {e})")

    # 保存图像网格
    if images:
        # 统一尺寸，避免拼接错位
        images = [im.resize((256, 256)) for im in images]
        grid = make_image_grid(images, cols=3)
        grid_path = os.path.join(SAVE_DIR, f"debug_{split}_samples.jpg")
        grid.save(grid_path)
        print(f"[DEBUG] 已保存 {split} 图像网格: {grid_path}")

    # 保存对应的文字
    txt_path = os.path.join(SAVE_DIR, f"debug_{split}_samples.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(captions))
    print(f"[DEBUG] 已保存 {split} 文本样本: {txt_path}")


# ========= 主流程 =========
def main():
    # 依赖 & 目录自检
    check_sentencepiece()
    assert_coco_layout(COCO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Processor（含图像预处理+分词）
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

    # 数据集
    train_ds = CocoCaptionDataset(COCO_ROOT, split="train", processor=processor, max_length=MAX_LENGTH)
    val_ds = CocoCaptionDataset(COCO_ROOT, split="val", processor=processor, max_length=MAX_LENGTH)

    print("Train size:", len(train_ds), " | Val size:", len(val_ds))

    # 可选：数据读取自检（导出图像网格与文本）
    if DEBUG_CHECK:
        debug_inspect_dataset(train_ds, "train")
        debug_inspect_dataset(val_ds, "val")

    # DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    print("Train batches:", len(train_loader), " | Val batches:", len(val_loader))

    # 模型 & 优化器 & 调度器
    model = SiglipDualEncoder("google/siglip-base-patch16-224").to(device)
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.2)
    total_steps = max(1, len(train_loader) * EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(1000, total_steps // 10),
                                                num_training_steps=total_steps)

    # 记录第一批的张量形状，帮助确认 batch 是否正确产生
    for first_batch in train_loader:
        pv = first_batch["pixel_values"]
        ids = first_batch["input_ids"]
        am = first_batch["attention_mask"]
        print(f"[DEBUG] 第一批张量形状 pixel_values={tuple(pv.shape)}, input_ids={tuple(ids.shape)}, attention_mask={tuple(am.shape)}")
        break

    # 日志文件
    log_file = os.path.join(SAVE_DIR, "train_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}  Train loss={train_loss:.4f} | Val loss={val_loss:.4f}")

        # 追加日志
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")

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
