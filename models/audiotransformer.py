import torch.nn as nn
from einops import rearrange
import torch
from functools import partial
from loguru import logger
from typing import Optional,List,Tuple
import torchaudio.transforms as audio_transforms
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import repeat

def perform_mixup_single(x: torch.Tensor, lamb: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """

    x1 = rearrange(x.flip(0), 'b ... -> ... b')
    x2 = rearrange(x.detach(), 'b ... -> ... b')
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')

def drop_patches(x: torch.Tensor, dim: int, frac: float) -> torch.Tensor:
    N = x.shape[dim]
    to_keep = N - int(N * frac)
    random_mask = torch.randperm(N, device=x.device)[:to_keep].sort().values
    return x.index_select(dim=dim, index=random_mask)


class NoNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias


class Normer(nn.Module):
    def __init__(self, mean, std, fac=2.0):
        super().__init__()
        self.mean = mean
        self.fac = fac
        self.std = std

    def forward(self, x):
        return (x - self.mean) / (self.fac * self.std)

class AudioPatchEmbed(nn.Module):
    def __init__(self,
                 input_size=224,
                 patch_size=16,
                 patch_stride=16,
                 in_chans=1,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=False):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = (input_size[0] // patch_stride[0],
                          input_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, 'b c f t -> b (f t) c')
        x = self.norm(x)
        return x

class cMlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)
        dim = q.shape[-1]

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * dim ** -0.5

        context = torch.einsum('bhnd,bhne->bhde', k, v)
        attn = torch.einsum('bhnd,bhde->bhne', q, context)
        return attn.reshape(*q.shape)


class LiteAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.to_q_attn_logits = nn.Linear(head_dim, 1, bias=False)
        self.to_k_attn_logits = nn.Linear(head_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_r = nn.Linear(head_dim, head_dim)
        self.to_out = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)
        q_aggr, k_aggr, v_aggr = q, k, v
        q_attn = torch.softmax(self.to_q_attn_logits(q) * self.scale, -1)
        global_q = (q_aggr * q_attn).sum(1, keepdim=True)

        k = k * global_q

        k_attn = torch.softmax(self.to_k_attn_logits(k) * self.scale, -1)
        global_k = (k_aggr * k_attn).sum(1, keepdim=True)

        u = v_aggr * global_k
        r = self.to_r(u)
        x = r + q
        x = self.proj_drop(x)
        x = x.reshape(B, N, C)
        return self.to_out(x)

class BNeckAttentionV2(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.inner_dim = self.num_heads

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, *_ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.inner_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BNeckAttentionV3(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.inner_dim = dim // 16

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, *_ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.inner_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BNeckAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.inner_dim = dim // 4

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, *_ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.inner_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 causal:bool = False,
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if mask is not None:
        # # Mask is a tensor of shape [B, T, T]
        # # Different from self.causal == True, the mask might be something like:
        # # [False, False, True]
        # # [False, False, True]
        # # [True, True, True]
        # # We use -inf to pad here, since if we would pad by any number, the entries at rows only containing
        # # [True, True, True] would lead to weights such as: [0.33,0.33,0.33], which is not correct
        # mask_value = torch.as_tensor(-float('inf'))
        # print(mask.shape, attn.shape)
        # attn = attn.masked_fill(mask, mask_value)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device,
                              dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RelativePosAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.pos_proj = nn.Linear(dim, dim, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor, pos_embedding:torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)
        # Q = (B, NH, T, D)
        pos_embedding = self.pos_proj(pos_embedding).view(
            B, -1, self.num_heads, self.head_dim)
        content_score = (q.transpose(1,2) + self.u_bias).transpose(1, 2) @ k.transpose(2, 3)
        pos_score = (q.transpose(1,2) + self.v_bias).transpose(1, 2) @ pos_embedding.permute(
            0, 2, 3, 1)
        pos_score = self._relative_shift(pos_score)

        attn = (pos_score + content_score) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class RelativeMultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias=False, attn_drop: float = 0., proj_drop=0.):
        super(RelativeMultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(dim)
        self.attention = RelativePosAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop, qkv_bias=qkv_bias)

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        outputs = self.attention(inputs,
                                 pos_embedding=pos_embedding,
                                 )
        return outputs


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_type='Attention',
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_type = globals()[attention_type]
        self.attn = attn_type(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class LinFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        seq_len,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        from linformer import LinformerSelfAttention
        self.attn = LinformerSelfAttention(
            dim=dim,
            seq_len=seq_len,
            k=8,
            dim_head=dim // num_heads // 4,
            heads=num_heads,
            dropout=attn_drop,
            one_kv_head=True,
            share_kv=True,
        )
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class AudioTransformer(nn.Module):
    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn:bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='token',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out:float = None,
                 freq_patch_out:float = None,
                 block_type='Block',
                 attention_type='Attention',
                 eval_avg = 'mean',
                 **kwargs
                 ):
        super().__init__()
        assert pooling in ('mean', 'token','dm')
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = kwargs.get('n_mels', 64)
        n_fft = kwargs.get('n_fft', 512)
        self.hop_size = kwargs.get('hop_size', 160)
        self.win_size = kwargs.get('win_size', 512)
        f_min = kwargs.get('f_min', 0)
        f_max = kwargs.get('f_max', 8000)
        self.center= kwargs.get('center', True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=self.win_size,
                                            center=self.center,
                                            n_fft=n_fft,
                                            f_max=f_max,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )

        if init_bn:
            self.init_bn = nn.Sequential(Rearrange('b c f t -> b f c t'),
                                         torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
                                         Rearrange('b f c t -> b c f t'))
        else:
            if self.n_mels == 64:
                self.init_bn = Normer(-10, 20)
            elif self.n_mels == 128:
                self.init_bn = Normer(-14.27, 20.79)
        self.target_length = target_length
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        self.spectransforms = nn.Sequential(
        ) if spectransforms is None else spectransforms
        self.wavtransforms = nn.Sequential() if wavtransforms is None else wavtransforms

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim))

        self.token_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * .02)
        self.time_pos_embed = nn.Parameter(
            torch.randn(1,embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        block_function = globals()[block_type]
        self.blocks = nn.Sequential(*[
            block_function(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  init_values=init_values,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  attention_type=attention_type,
                  ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, outputdim))
        self.apply(self.init_weights)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'}

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


    def forward_features(self, x):
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed
        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        if self.pooling == 'token':
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'mean':
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'dm':
            # Unpack using the frequency dimension, which is constant
            x = rearrange(x,
                          'b (f t) d -> b f t d',
                          f=self.patch_embed.grid_size[0])
            #First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)



    def load_state_dict(self, state_dict, strict=True):
        if 'time_pos_embed' in state_dict and self.time_pos_embed.shape != state_dict[
                'time_pos_embed'].shape:
            logger.debug(
                "Positional Embedding shape not the same with model, resizing!"
            )
            self.change_pos_embedding(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward(self, x, mixup=None):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if self.training and mixup != None:
            x = perform_mixup_single(x, mixup)
        if self.training:
            x = self.spectransforms(x)
        x = rearrange(x, 'b f t -> b 1 f t')
        if self.init_bn is not None:
            x = self.init_bn(x)
        if not self.training and x.shape[-1] > self.target_length:
            #When testing with a longer input
            outs = []
            # splits = x.unfold(-1, self.target_length,
            # 16).permute(3, 0, 1, 2, 4)
            # for f in splits:
            # Just drop the last sample, enhances performance
            for f in x.split(self.target_length, -1):
                # Pad zeros for usually last chunk, that we have self.target_length assured
                if f.shape[-1] != self.target_length:
                    continue
                    # pad = torch.zeros((*x.shape[:-1], self.target_length),
                                      # device=x.device)
                    # pad[..., :f.shape[-1]] = f
                    # f = pad
                outs.append(self.forward_head(self.forward_features(f)))
            x = torch.stack(outs,-1)
            if self.eval_avg == 'mean':
                x = x.mean(-1)
            elif self.eval_avg == 'max':
                x = x.max(-1)[0]
            else:
                raise ValueError(f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        return x, None

class AudioTransformer_Pool(AudioTransformer):
    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn:bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='token',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out:float = None,
                 freq_patch_out:float = None,
                 pool_idxs:List[int] = [1], # Index of where to pool
                 block_type='Block',
                 attention_type='Attention',
                 eval_avg = 'mean',
                 **kwargs):
        super().__init__(
                 outputdim=outputdim,
                 patch_size=patch_size,
                 patch_stride=patch_stride,
                 embed_dim=embed_dim,
                 depth=depth,
                 num_heads=num_heads,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 init_bn = init_bn,
                 norm_layer=norm_layer,
                 act_layer=act_layer,
                 init_values=init_values,
                 target_length=target_length,
                 pooling=pooling,
                 wavtransforms=wavtransforms,
                 spectransforms=spectransforms,
                 time_patch_out = time_patch_out,
                 freq_patch_out = freq_patch_out,
                 block_type=block_type,
                 attention_type=attention_type,
                 eval_avg =eval_avg,
                **kwargs)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        block_function = globals()[block_type]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(block_function(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  init_values=init_values,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  attention_type=attention_type,
                  ))
            if i in pool_idxs:
                blocks.append(
                    nn.Sequential(
                        Rearrange('b t d -> b d t'),
                        nn.AvgPool1d(2),
                        Rearrange('b d t -> b t d'),
                    ))
        self.blocks = nn.Sequential(*blocks)

class AudioTransformer_LinFormer(AudioTransformer):
    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn:bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='token',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out:float = None,
                 freq_patch_out:float = None,
                 block_type='Block',
                 eval_avg = 'mean',
                 **kwargs):
        super().__init__(
                 outputdim=outputdim,
                 patch_size=patch_size,
                 patch_stride=patch_stride,
                 embed_dim=embed_dim,
                 depth=depth,
                 num_heads=num_heads,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 init_bn = init_bn,
                 norm_layer=norm_layer,
                 act_layer=act_layer,
                 init_values=init_values,
                 target_length=target_length,
                 pooling=pooling,
                 wavtransforms=wavtransforms,
                 spectransforms=spectransforms,
                 time_patch_out = time_patch_out,
                 freq_patch_out = freq_patch_out,
                 block_type='Block', # Will be overwritten
                 attention_type='Attention', # Will be overwritten
                 eval_avg =eval_avg,
                **kwargs)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        block_function = globals()[block_type]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(
                LinFormerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    seq_len=int(self.patch_embed.grid_size[0] *
                                self.patch_embed.grid_size[1]),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ))
        self.blocks = nn.Sequential(*blocks)


class AudioTransformer_Relative(AudioTransformer):
    def __init__(self, *args, **kwargs ):
        super().__init__(attention_type='RelativeMultiHeadedSelfAttentionModule',*args, **kwargs)
        del self.token_pos_embed,self.freq_pos_embed,self.time_pos_embed

    def forward_features(self, x):
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.cls_token is not None and self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


def audio_transformer_v1(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=1,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_pool_v1(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=1,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_Pool(**model_kwargs)

def audio_transformer_tiny_d6(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        init_bn=True,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h192_d6_m4(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=1,
        init_bn=True,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h256_d1_m4(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=256,
        depth=1,
        num_heads=1,
        init_bn=True,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_relu_nonorm(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        norm_layer = NoNorm,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_bneck_v3_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttentionV3',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_bneck_v2_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttentionV2',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d4_m3_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)


def audio_transformer_h128_d4_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m3_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m3_token(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='token',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d2_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=2,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d12_m3_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d12_m3_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d12_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h768_d6_m4(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=12,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_base_fb(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_huge_fb(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_large_fb(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_tiny(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d4_m3_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=4,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d5_m3_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=5,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m2_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=2,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m3_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d6_m3_relu_linformer(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_LinFormer(**model_kwargs)

def audio_transformer_h128_d3_m3_bneck_relu_lin(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='LinearAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_Pool(**model_kwargs)

def audio_transformer_h128_d3_m3_bneck_relu_pool(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_Pool(**model_kwargs)

def audio_transformer_h128_d6_m3_bneck_relu_pool(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_Pool(**model_kwargs)

def audio_transformer_h128_d3_m3_h1_bneck_relu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=1,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.ReLU,
        attention_type='BNeckAttention',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3_silu(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0,
        act_layer=nn.SiLU,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h128_d3_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=3.0,
        pooling='mean',
        init_bn=True,
        drop_path_rate=0.0
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def audio_transformer_h96_d4_m4(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=96,
        depth=4,
        num_heads=2,
        mlp_ratio=4,
        pooling='mean',
        init_bn=True
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)

def relative_audio_transformer_h128_d3_m3(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=128,
        depth=3,
        num_heads=2,
        init_bn=True,
        mlp_ratio=3,
        pooling='mean',
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer_Relative(**model_kwargs)


def audio_transformer_s(**kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_bn=True,
        mlp_ratio=4,
    )
    model_kwargs.update(
        (k, kwargs[k]) for k in set(kwargs).intersection(model_kwargs))
    model_kwargs = {**model_kwargs, **kwargs}
    return AudioTransformer(**model_kwargs)


if __name__ == "__main__":
    from torchinfo import summary
    # att = LiteAttention(128, num_heads=1)
    # x = torch.randn(1, 100, 128)
    # att(x)
    # trans = audio_transformer_h128_d3_m3_relu_nonorm(target_length=200)
    # trans = audio_transformer_h128_d4_m3_bneck_relu(target_length=201)
    trans = audio_transformer_h128_d6_m2_bneck_relu(target_length=201)
    x = torch.randn(4, 32000)
    x1 = trans.front_end(x)
    trans.eval()
    summary(trans, input_size=x.shape, device='cpu')
    y, _ = trans(x)
    torch.save(trans.state_dict(), '/tmp/mdl.pt')
    # print(y.shape)
