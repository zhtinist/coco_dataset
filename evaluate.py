import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from dataset import COCODataset, get_dataloader  # 使用你提供的 dataset.py 文件

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型（使用CLIP模型）
model_name = "openai/clip-vit-base-patch32"  # 使用CLIP模型
vision_model = CLIPModel.from_pretrained(model_name).vision_model.to(device)
text_model = CLIPModel.from_pretrained(model_name).text_model.to(device)

# 加载图像处理器
processor = CLIPProcessor.from_pretrained(model_name)

@torch.no_grad()
def evaluate(model_text, model_vision, dataloader, threshold=0.5, device="cuda"):
    model_text.eval()
    model_vision.eval()

    all_labels, all_scores = [], []
    for batch in tqdm(dataloader):
        pixel_values, input_ids, attention_mask = batch
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 获取图像和文本嵌入
        image_embeds = model_vision(pixel_values=pixel_values).pooler_output
        text_embeds = model_text(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # 归一化嵌入
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

        # 计算相似度分数
        scores = (image_embeds @ text_embeds.T).diag().cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend([1] * len(scores))  # 假设为正例

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Precision@threshold 和 Recall@threshold
    preds = (all_scores > threshold).astype(int)
    precision = (preds[all_labels == 1].sum()) / max(preds.sum(), 1)
    recall = (preds[all_labels == 1].sum()) / (all_labels == 1).sum()

    # PR-AUC
    precisions, recalls, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = np.trapz(precisions, recalls)

    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_scores)

    # MRR (Mean Reciprocal Rank)
    ranks = np.argsort(-all_scores)  # 排序
    mrr = np.mean([1 / (np.where(ranks == i)[0][0] + 1) for i, l in enumerate(all_labels) if l == 1])

    return {"precision": precision, "recall": recall, "pr_auc": pr_auc, "roc_auc": roc_auc, "mrr": mrr}

# 创建COCO数据集和数据加载器
dataset = COCODataset(root=r"D:\datasets\coco", split="val")  # 修改为实际路径
dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=4)

# 执行评估
metrics = evaluate(text_model, vision_model, dataloader, device=device)
print(metrics)
