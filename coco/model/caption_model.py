import torch
from torch import nn

from transformers import SiglipVisionModel, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


class COCOCaptionModel(nn.Module):
    def __init__(self):
        super(COCOCaptionModel, self).__init__()
        self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
        self.decoder = GPT2LMHeadModel.from_pretrained("distilgpt2")

    def forward(self, image_tensors, input_ids, labels=None):
        # 1. get vision embeddings
        vision_embeds = self.vision_encoder(image_tensors).last_hidden_state  # N * vision_seq_len * 768

        # 2. get text embeddings
        text_embeds = self.decoder.transformer.wte(input_ids)  # N * text_seq_len * 768

        # 3. concat
        fused_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # N * (vision_seq_len + text_seq_len) * 768

        if labels is not None:
            batch_size, vision_seq_len, _ = vision_embeds.shape
            fused_labels = torch.cat([
                torch.ones([batch_size, vision_seq_len], dtype=labels.dtype, device=labels.device) * -100, labels
            ], dim=1)  # N * (vision_seq_len + text_seq_len)

        # 4. decoding
        outputs = self.decoder(inputs_embeds=fused_embeds, labels=fused_labels)
        return outputs

    def generate(self, image_tensors):
        # 1. tokenize system prompt
        system_prompt = "What is inside this image?"
        prompt_tokens = self.tokenizer(system_prompt, return_tensors="pt")

        # 2. get vision embeddings
        vision_embeds = self.vision_encoder(image_tensors).last_hidden_state  # N * vision_seq_len * 768

        # 3. get text embeddings
        text_embeds = self.decoder.transformer.wte(prompt_tokens.input_ids)  # N * text_seq_len * 768

        # 4. fusion
        fused_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # N * (vision_seq_len + text_seq_len) * 768

        # 5. generate
        outputs = self.decoder.generate(inputs_embeds=fused_embeds)

        # 6. decode
        return tokenizer.batch_decode(outputs)


if __name__ == "__main__":
    import torch
    image_tensors = torch.rand([2, 3, 224, 224])
    input_ids = torch.randint(0, 10, [2, 32])
    labels = input_ids.clone()

    model = COCOCaptionModel()
    outputs = model(image_tensors, input_ids, labels)
    print(f"loss: {outputs.loss}")
