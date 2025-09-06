from transformers import AutoModel, AutoProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score
from PIL import Image
import os
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/siglip2-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
vision_model = model.vision_model.to(device)
text_model = model.text_model.to(device)

class COCODataset(Dataset):
    def __init__(self, root, split='train', processor=None):
        self.root = root
        self.split = split
        self.processor = processor or AutoProcessor.from_pretrained(model_name)

        ann_file = os.path.join(root, f"annotations/captions_{split}2017.json")
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)['annotations']

        self.image_dir = os.path.join(root, f"{split}2017")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id, caption = ann['image_id'], ann['caption']
        image_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Warning] Skip missing file: {image_path}")
            return None

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = torch.ones_like(input_ids)

        return inputs['pixel_values'], input_ids, attention_mask

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    pixel_list, id_list, mask_list = zip(*batch)

    id_list_clean, mask_list_clean = [], []
    for ids, mask in zip(id_list, mask_list):
        ids = ids.squeeze().tolist()
        mask = mask.squeeze().tolist()

        # 强制 flatten
        if isinstance(ids[0], list): ids = ids[0]
        if isinstance(mask[0], list): mask = mask[0]

        # 强制对齐长度
        if len(mask) != len(ids):
            mask = [1] * len(ids)

        id_list_clean.append(ids)
        mask_list_clean.append(mask)

    max_len = max(len(x) for x in id_list_clean)

    # 手动 pad
    padded_ids = torch.full((len(id_list_clean), max_len), processor.tokenizer.pad_token_id, dtype=torch.long)
    padded_mask = torch.zeros((len(mask_list_clean), max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(id_list_clean, mask_list_clean)):
        padded_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded_mask[i, :len(mask)] = torch.tensor(mask, dtype=torch.long)

    pixel_values = torch.cat(pixel_list, dim=0)

    print(f"[DEBUG] Collated batch size={len(id_list_clean)}, max_len={max_len}")
    return pixel_values, padded_ids, padded_mask

def get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn)

@torch.no_grad()
def evaluate(model_text, model_vision, dataloader, threshold=0.5, device="cuda"):
    model_text.eval()
    model_vision.eval()

    all_labels, all_scores = [], []
    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue

        pixel_values, input_ids, attention_mask = batch
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        image_embeds = model_vision(pixel_values=pixel_values).pooler_output
        text_embeds = model_text(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        scores = (image_embeds @ text_embeds.T).diag().cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend([1] * len(scores))

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    preds = (all_scores > threshold).astype(int)
    precision = preds[all_labels == 1].sum() / max(preds.sum(), 1)
    recall = preds[all_labels == 1].sum() / (all_labels == 1).sum()

    precisions, recalls, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = np.trapz(precisions, recalls)
    roc_auc = roc_auc_score(all_labels, all_scores)

    ranks = np.argsort(-all_scores)
    mrr = np.mean([1.0 / (np.where(ranks == i)[0][0] + 1)
                   for i, l in enumerate(all_labels) if l == 1])

    return {"precision": precision, "recall": recall,
            "pr_auc": pr_auc, "roc_auc": roc_auc, "mrr": mrr}

if __name__ == "__main__":
    dataset = COCODataset(root=r"D:\\datasets\\coco", split="val", processor=processor)
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)

    metrics = evaluate(text_model, vision_model, dataloader, device=device)
    print(metrics)
