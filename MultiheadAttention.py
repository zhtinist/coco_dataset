import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.head_dim = embed_dim // num_heads

        # projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        
        # output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        
        # optional bias for key/value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True):
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)  # (L, N, E)

        L, N, E = query.shape
        S = key.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, N, 1)], dim=0)
            v = torch.cat([v, self.bias_v.repeat(1, N, 1)], dim=0)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = F.pad(attn_mask, (0, 1))
                elif attn_mask.dim() == 3:
                    attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        q = q.contiguous().view(L, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N*h, L, head_dim)
        k = k.contiguous().view(-1, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N*h, S, head_dim)
        v = v.contiguous().view(-1, N * self.num_heads, self.head_dim).transpose(0, 1)  # (N*h, S, head_dim)

        if self.add_zero_attn:
            k = torch.cat([k, torch.zeros((N * self.num_heads, 1, self.head_dim), device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((N * self.num_heads, 1, self.head_dim), device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  # (N*h, L, S)

        # ----- mask 处理 -----
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # (1, L, S)
            if attn_mask.size(0) == 1:
                attn_mask = attn_mask.expand(N * self.num_heads, -1, -1)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            key_padding_mask = key_padding_mask.reshape(N * self.num_heads, 1, S)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)  # (N*h, L, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # (N, L, E)

        if need_weights:
            attn_output_weights = attn_output_weights.view(N, self.num_heads, L, S)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


def generate_encoder_mask(seq_len, device=None):
    return torch.zeros(seq_len, seq_len, device=device)

def generate_decoder_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


if __name__ == "__main__":
    torch.manual_seed(42)

    embed_dim = 768
    num_heads = 8
    batch_size = 2
    seq_len = 6

    # 官方 MHA
    off_mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    # 自定义 MHA
    my_mha = CustomMultiheadAttention(embed_dim, num_heads, batch_first=True)

    # 同步参数
    my_mha.q_proj.weight.data = off_mha.in_proj_weight.data[:embed_dim, :].clone()
    my_mha.k_proj.weight.data = off_mha.in_proj_weight.data[embed_dim:2*embed_dim, :].clone()
    my_mha.v_proj.weight.data = off_mha.in_proj_weight.data[2*embed_dim:, :].clone()

    my_mha.q_proj.bias.data = off_mha.in_proj_bias.data[:embed_dim].clone()
    my_mha.k_proj.bias.data = off_mha.in_proj_bias.data[embed_dim:2*embed_dim].clone()
    my_mha.v_proj.bias.data = off_mha.in_proj_bias.data[2*embed_dim:].clone()

    my_mha.out_proj.weight.data = off_mha.out_proj.weight.data.clone()
    my_mha.out_proj.bias.data = off_mha.out_proj.bias.data.clone()

    x = torch.randn(batch_size, seq_len, embed_dim)

    print("\n===== Encoder (双向) =====")
    enc_mask = generate_encoder_mask(seq_len)
    my_out, _ = my_mha(x, x, x, attn_mask=enc_mask)
    off_out, _ = off_mha(x, x, x, attn_mask=enc_mask)
    print("是否数值一致:", torch.allclose(my_out, off_out, atol=1e-6))
    print("最大绝对误差:", (my_out - off_out).abs().max().item())

    print("\n===== Decoder (因果) =====")
    dec_mask = generate_decoder_mask(seq_len)
    my_out, _ = my_mha(x, x, x, attn_mask=dec_mask)
    off_out, _ = off_mha(x, x, x, attn_mask=dec_mask)
    print("是否数值一致:", torch.allclose(my_out, off_out, atol=1e-6))
    print("最大绝对误差:", (my_out - off_out).abs().max().item())
