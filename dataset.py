import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPImageProcessor


# ---------- collate_fn ----------
def collate_fn(batch):
    image_tensors, input_ids, attention_mask = zip(*batch)
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], dim=0)
    input_ids = torch.cat([t.unsqueeze(0) for t in input_ids], dim=0)
    attention_mask = torch.cat([t.unsqueeze(0) for t in attention_mask], dim=0)
    return image_tensors, input_ids, attention_mask


# ---------- DataLoader 封装 ----------
def get_dataloader(dataset,
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   drop_last: bool = True) -> DataLoader:
    """
    根据 COCODataset 构建 DataLoader。

    Args:
        dataset (COCODataset): 已实例化的 COCODataset。
        batch_size (int): 批次大小。
        shuffle (bool, optional): 是否打乱数据。默认 True。
        num_workers (int, optional): 子进程加载数据数量。默认 0。
        drop_last (bool, optional): 是否丢弃最后一个不完整的批次。默认 True。

    Returns:
        DataLoader: 配置好的 PyTorch DataLoader。
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=True
    )


# ---------- COCO 数据集 ----------
class COCODataset(Dataset):
    def __init__(self,
                 root: str = r"D:/datasets/coco",
                 split: str = "train",
                 max_length: int = 77):      # CLIP 文本最大长度
        super().__init__()
        self.root = root
        self.split = split
        self.max_length = max_length

        # 1. 读取标注文件
        anno_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
        with open(anno_file, encoding="utf-8") as f:
            anno = json.load(f)

        # image_id -> [captions]
        img2caps = {}
        for cap in anno["annotations"]:
            img_id = cap["image_id"]
            img2caps.setdefault(img_id, []).append(cap["caption"])

        # image_id -> file_name
        id2fname = {img["id"]: img["file_name"] for img in anno["images"]}

        # 2. 构造样本列表：[(image_id, caption), ...]
        self.samples = [(img_id, cap)
                        for img_id, caps in img2caps.items()
                        for cap in caps]
        self.id2fname = id2fname

        # 3. CLIP tokenizer & image_processor
        model_name = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        fname = self.id2fname[img_id]

        img_path = os.path.join(self.root, f"{self.split}2017", fname)
        image = Image.open(img_path).convert("RGB")

        # CLIP 官方图像预处理
        pixel_values = self.image_processor(
            images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # CLIP 官方文本编码
        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return pixel_values, input_ids, attention_mask


# ---------- 本地测试 ----------
if __name__ == "__main__":
    ds = COCODataset(root=r"C:\Users\61556\Downloads\data\coco", split="val")
    print("Dataset size:", len(ds))

    img, ids, mask = ds[0]
    print("Image shape:", img.shape)
    print("Token ids shape:", ids.shape)
    print("Attention mask shape:", mask.shape)

    loader = get_dataloader(ds, batch_size=4, shuffle=True)
    for imgs, ids, mask in loader:
        print("Batch image tensors shape:", imgs.shape)
        print("Batch input IDs shape:", ids.shape)
        print("Batch attention mask shape:", mask.shape)
        break