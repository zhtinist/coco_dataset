import torch


def multi_head_attention(query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Function to reproduct the `MultiheadAttention` Layer in Pytorch library
    """
    raise NotImplementedError


if __name__ == "__main__":
    query = torch.randn([4, 512, 768])
    key = value = torch.randn([4, 128, 768])
    attn = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
    attn_params = attn.state_dict()
    output_1, _ = attn(query, key, value)
    output_2 = multi_head_attention(
        query, key, value,
        in_proj_weight=attn_params["in_proj_weight"],
        in_proj_bias=attn_params["in_proj_bias"],
        out_proj_weight=attn_params["out_proj.weight"],
        out_proj_bias=attn_params["out_proj.bias"]
    )
    max_diff = (output_1 - output_2).abs().max()
    assert max_diff < 1e-4
