import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

class COCODataset(Dataset):
    def __init__(self,
                 root: str = r"D:\\datasets\\coco\\train2017",  # <<< 改这里
                 split: str = "train",
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 32):
        super().__init__()
        self.root = root
        self.split = split
        self.max_length = max_length

        # 1. 读取标注
        anno_file = os.path.join(root, "annotations", f"captions_{split}2017.json")
        with open(anno_file, encoding="utf-8") as f:
            anno = json.load(f)

        # image_id -> [captions]
        img2caps = {}
        for cap in anno["annotations"]:
            img_id = cap["image_id"]
            img2caps.setdefault(img_id, []).append(cap["caption"])

        # image_id -> file_name
        id2fname = {img["id"]: img["file_name"]
                    for img in anno["images"]}

        # 2. 构造样本列表
        self.samples = [(img_id, cap) for img_id, caps in img2caps.items() for cap in caps]
        self.id2fname = id2fname

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 3. 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        fname = self.id2fname[img_id]

        img_path = os.path.join(self.root, f"{self.split}2017", fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        enc = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask



if __name__ == "__main__":
    ds = COCODataset(split="train")
    print("Dataset size:", len(ds))
    img, ids, mask = ds[0]
    print("Image shape:", img.shape)
    print("Token ids shape:", ids.shape)
    print("Attention mask shape:", mask.shape)