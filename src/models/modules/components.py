import torch
from torch import nn
from torch.nn import functional as F
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention (Manual implementation commented out for optimization)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # attn_weights = F.softmax(scores, dim=-1)
        # attn_output = torch.matmul(attn_weights, v)

        # Optimized attention computation using PyTorch's built-in function
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.linear_out(attn_output)
        
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.gate_linear = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU()
        )
        self.up_linear = nn.Linear(d_model, d_ff, bias=False)
        self.down_linear = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate_output = self.gate_linear(x)
        up_output = self.up_linear(x)
        ff_output = gate_output * up_output
        output = self.down_linear(ff_output)
        return output

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x
    
if __name__ == "__main__":
    # Test the components with dummy data
    batch_size = 2
    seq_length = 36
    d_model = 32
    num_heads = 4
    d_ff = 64

    x = torch.randn(batch_size, seq_length, d_model)

    block = Block(d_model, num_heads, d_ff)
    output = block(x)
    print("Output shape:", output.shape)  # Should be (batch_size, seq_length, d_model)