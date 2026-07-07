import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FiLM1D(nn.Module):
    """FiLM từ code gốc — dùng cho cond (B,T,C)"""
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.Linear(cond_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )

    def forward(self, x, cond):  # x:(B,T,D), cond:(B,T,C) hoặc (B,C)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)
        s, b = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + s) + b


class FiLM_Time(nn.Module):
    """FiLM đơn giản cho time embedding (B, time_dim) → scale/shift"""
    def __init__(self, time_dim, hidden):
        super().__init__()
        self.to_scale_shift = nn.Linear(time_dim, hidden * 2)

    def forward(self, x, temb):  # x:(B,T,C), temb:(B,time_dim)
        scale, shift = self.to_scale_shift(temb).chunk(2, dim=-1)
        return x * (1 + scale[:, None]) + shift[:, None]


class hybridST_ATTENTION_1D(nn.Module):
    """
    Chia channel C thành 2 nửa:
      - Nhánh 's' (spatial/local): depthwise Conv1d + self-attention trên T
      - Nhánh 't' (temporal/global): depthwise Conv1d + self-attention trên T
    Concat → proj → residual.
    Input/output: (B, T, C)
    """
    def __init__(self, d_coor, head=8):
        super().__init__()
        assert d_coor % (head * 2) == 0, \
            f"d_coor ({d_coor}) phải chia hết cho head*2 ({head*2})"
        self.head = head
        self.scale = (d_coor // 2 // head) ** -0.5

        self.layer_norm = nn.LayerNorm(d_coor)
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.proj = nn.Linear(d_coor, d_coor)

        # Depthwise Conv1d — tương đương sep2_s / sep2_t trong hybridST gốc
        self.sep_s = nn.Conv1d(d_coor // 2, d_coor // 2,
                               kernel_size=3, stride=1, padding=1,
                               groups=d_coor // 2)
        self.sep_t = nn.Conv1d(d_coor // 2, d_coor // 2,
                               kernel_size=3, stride=1, padding=1,
                               groups=d_coor // 2)

    def forward(self, x):
        """x: (B, T, C)"""
        B, T, C = x.shape
        h = x                              # residual

        x = self.layer_norm(x)
        qkv = self.qkv(x)                 # (B, T, C*3)
        qkv = qkv.reshape(B, T, C, 3).permute(3, 0, 1, 2)   # (3, B, T, C)

        # Chia channel → nhánh s và nhánh t
        qkv_s, qkv_t = qkv.chunk(2, dim=-1)                  # each: (3, B, T, C//2)

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]        # (B, T, C//2)
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]

        # --- Nhánh s: conv trên chiều T rồi attention ---
        v_s = self.sep_s(v_s.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, C//2)

        q_s = rearrange(q_s, 'b t (h c) -> (b h) t c', h=self.head)
        k_s = rearrange(k_s, 'b t (h c) -> (b h) c t', h=self.head)
        v_s = rearrange(v_s, 'b t (h c) -> (b h) t c', h=self.head)

        att_s = (q_s @ k_s) * self.scale          # (B*H, T, T)
        att_s = att_s.softmax(-1)
        x_s = att_s @ v_s                          # (B*H, T, c_head)
        x_s = rearrange(x_s, '(b h) t c -> b t (h c)', h=self.head)  # (B, T, C//2)

        # --- Nhánh t: conv trên chiều T rồi attention ---
        v_t = self.sep_t(v_t.permute(0, 2, 1)).permute(0, 2, 1)

        q_t = rearrange(q_t, 'b t (h c) -> (b h) t c', h=self.head)
        k_t = rearrange(k_t, 'b t (h c) -> (b h) c t', h=self.head)
        v_t = rearrange(v_t, 'b t (h c) -> (b h) t c', h=self.head)

        att_t = (q_t @ k_t) * self.scale
        att_t = att_t.softmax(-1)
        x_t = att_t @ v_t
        x_t = rearrange(x_t, '(b h) t c -> b t (h c)', h=self.head)  # (B, T, C//2)

        # --- Ghép 2 nhánh ---
        x = torch.cat([x_s, x_t], dim=-1)         # (B, T, C)
        x = self.proj(x)
        return x + h


class hybridST_BLOCK_1D(nn.Module):
    """
    Tương đương hybridST_BLOCK nhưng có thêm FiLM conditioning
    cho cả cond (B,T,cond_dim) lẫn time embedding (B,time_dim).
    """
    def __init__(self, d_coor, cond_dim, time_dim, head=8):
        super().__init__()
        self.att       = hybridST_ATTENTION_1D(d_coor, head=head)
        self.film_cond = FiLM1D(cond_dim, d_coor)
        self.film_time = FiLM_Time(time_dim, d_coor)
        self.layer_norm = nn.LayerNorm(d_coor)
        self.mlp        = Mlp(d_coor, d_coor, d_coor)

    def forward(self, x, cond, temb):
        """
        x    : (B, T, C)
        cond : (B, T, cond_dim)
        temb : (B, time_dim)
        """
        x = self.att(x)                          # hybrid ST attention + residual
        x = self.film_cond(x, cond)              # condition từ encoder
        x = self.film_time(x, temb)              # condition từ timestep
        x = x + self.mlp(self.layer_norm(x))     # MLP + residual
        return x


class AttentionDenoiser(nn.Module):
    """
    Denoiser theo cấu trúc hybridSTFormer.
    Đầu vào / đầu ra giữ nguyên: (y_noisy, cond, t) → (eps_hat, feat)
    """
    def __init__(self, y_dim=1, cond_dim=128, hidden=256, depth=8,
                 time_dim=128, feat_out=128, head=8):
        super().__init__()

        # Time embedding — giống hybridST dùng positional param
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Positional embedding theo chiều thời gian (tương đương temporal_pos_embedding)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 1, hidden))

        # Input projections
        self.in_y = nn.Linear(y_dim, hidden)
        self.in_c = nn.Linear(cond_dim, hidden)

        # Stack hybridST_BLOCK_1D — tương đương stc_block trong hybridSTFormer
        self.stc_block = nn.ModuleList([
            hybridST_BLOCK_1D(
                d_coor=hidden,
                cond_dim=hidden,
                time_dim=time_dim,
                head=head
            )
            for _ in range(depth)
        ])

        # Output heads — tương đương regress_head trong hybridST
        self.out_norm = nn.LayerNorm(hidden)
        self.out_eps  = nn.Linear(hidden, y_dim)
        self.out_feat = nn.Linear(hidden, feat_out)

    def forward(self, y_noisy, cond, t):
        """
        y_noisy : (B, T, y_dim)
        cond    : (B, T, cond_dim)
        t       : (B,) long
        Returns : eps_hat (B,T,y_dim), feat (B,T,feat_out)
        """
        temb = self.time_mlp(t)                        # (B, time_dim)

        hy = self.in_y(y_noisy)                        # (B, T, hidden)
        hc = self.in_c(cond)                           # (B, T, hidden)

        # Positional embedding + fusion (tương đương forward của hybridSTFormer)
        h = hy + hc + self.temporal_pos_embedding      # (B, T, hidden)

        # Qua từng hybridST block
        for blk in self.stc_block:
            h = blk(h, cond=hc, temb=temb)

        h = self.out_norm(h)
        eps_hat = self.out_eps(h)                      # (B, T, y_dim)
        feat    = self.out_feat(h)                     # (B, T, feat_out)
        return eps_hat, feat


def cosine_alphas_cumprod(T, s=0.008, device="cpu"):
    t = torch.linspace(0, T, T + 1, device=device, dtype=torch.float32)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    f = f / f[0]
    abar = f[1:]
    alphas = abar / torch.cat([torch.tensor([1.0], device=device, dtype=torch.float32), abar[:-1]])
    betas = (1. - alphas).clamp(1e-6, 0.999)
    return betas, alphas, abar


class DiffMEBackBone(nn.Module):
    def __init__(self, cond_dim=128, y_dim=1, steps=1500, schedule='cosine',
                 hidden=256, time_dim=128, feat_out=128, depth=8,
                 guidance_scale=1.5, head=8):
        super().__init__()
        self.inference_timesteps = [0, 150, 300, 500, 700]
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.cond_drop_prob = 0.0

        if schedule == 'cosine':
            betas, alphas, abar = cosine_alphas_cumprod(steps)
        else:
            betas = torch.linspace(1e-4, 2e-2, steps, dtype=torch.float32)
            alphas = 1. - betas
            abar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('abar', abar)

        self.denoiser = AttentionDenoiser(
            y_dim=y_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            time_dim=time_dim,
            feat_out=feat_out,
            depth=depth,
            head=head,
        )

    def q_sample(self, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        abar_t = self.abar[t].view(-1, 1, 1)
        return abar_t.sqrt() * y0 + (1 - abar_t).sqrt() * noise, noise

    def forward(self, cond, y_star=None):
        B, T, _ = cond.shape
        device = cond.device

        if y_star is not None:
            if y_star.dim() == 2:
                y_star = y_star.unsqueeze(-1).float()
            elif y_star.dim() == 3 and y_star.size(-1) != 1:
                raise ValueError("y_star phải có shape (B,T) hoặc (B,T,1)")
            t = torch.randint(0, self.steps, (B,), device=device, dtype=torch.long)
            y_t, noise = self.q_sample(y_star, t)
            eps_hat, feat = self.denoiser(y_t, cond, t)
            diff_loss = F.mse_loss(eps_hat, noise)
            aux = {"t": t, "y_t": y_t, "noise": noise, "eps_hat": eps_hat}
            return diff_loss, feat, aux
        else:
            feats = []
            for ts in self.inference_timesteps:
                t = torch.full((B,), ts, device=device, dtype=torch.long)
                y_zeros = torch.zeros(B, T, 1, device=device, dtype=cond.dtype)
                _, feat_t = self.denoiser(y_zeros, cond, t)
                feats.append(feat_t)
            feat = torch.stack(feats, dim=0).mean(dim=0)
            aux = {"t": torch.tensor(self.inference_timesteps, device=device)}
            return None, feat, aux


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DeformConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=True)
        self.offset_conv = nn.Conv1d(in_channels, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.offset_conv.weight)

    def forward(self, x):
        out = self.conv(x)
        offset = self.offset_conv(x).mean(dim=1, keepdim=True)
        out = out * (1.0 + 0.005 * torch.tanh(offset))
        return out


class MSBlock(nn.Module):
    def __init__(self, d_model, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d)
            for d in dilations
        ])
        self.proj   = nn.Conv1d(d_model * len(dilations), d_model, 1)
        self.deform = DeformConv1d(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_in = x
        feats = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat(feats, dim=1)
        x = self.proj(x)
        x = self.deform(x)
        x = self.dropout(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + x_in * 1.1


class MultiScaleSpotterHead(nn.Module):
    def __init__(self, d_model, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([MSBlock(d_model) for _ in range(num_blocks)])
        self.out    = nn.Conv1d(d_model, 1, kernel_size=1)

    def forward(self, feat):  # feat: (B, d_model, T)
        x = feat
        for blk in self.blocks:
            x = blk(x)
        logits = self.out(x).squeeze(1)   # (B, T)
        prob   = torch.sigmoid(logits)
        return prob, logits



class DiffME(nn.Module):
    def __init__(self, out_channels=5, dim=128, feat_out=128,
                 diff_hidden=256, time_dim=128, depth=8,
                 guidance_scale=1.5, use_frd=False,
                 enable_recognition=True, head=8):
        super().__init__()
        self.enable_recognition = enable_recognition
        self.out_channels = out_channels
        self.use_frd = use_frd

        # Stem
        self.Stem = nn.Sequential(
            nn.Conv1d(36, dim, 3, 1, 1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
        )

        self.diffusion = DiffMEBackBone(
            cond_dim=dim, y_dim=1, steps=1000, schedule='cosine',
            hidden=diff_hidden, time_dim=time_dim, feat_out=feat_out,
            depth=depth, guidance_scale=guidance_scale, head=head,
        )

        # Mamba stack
        self.mamba_stack = nn.ModuleList([
            Mamba(d_model=feat_out, d_state=32, d_conv=4, expand=2)
            for _ in range(6)
        ])
        self.mamba_norm = nn.LayerNorm(feat_out)

        # Spotting path
        self.feat_proj  = nn.Linear(feat_out, 16)
        self.spot_head  = MultiScaleSpotterHead(d_model=16, num_blocks=4)
        self.skip_proj  = nn.Linear(feat_out, 16)
        self.skip_scale = nn.Parameter(torch.tensor(0.4))

        # Recognition path
        if self.enable_recognition:
            self.cond_proj  = nn.Linear(dim, 64)
            self.ring_norm  = nn.LayerNorm(feat_out + 64)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feat_out + 64, num_heads=8,
                dropout=0.1, batch_first=True
            )
            self.recog_head = nn.Sequential(
                nn.Conv1d(feat_out + 64, 256, 3, 1, 1),
                nn.BatchNorm1d(256), nn.SiLU(),
                nn.Conv1d(256, 256, 3, 1, 1),
                nn.BatchNorm1d(256), nn.SiLU()
            )
            self.fc = nn.Linear(256, out_channels)
        else:
            self.cond_proj = self.ring_norm = self.cross_attn = None
            self.recog_head = self.fc = None

        self.apply(segm_init_weights)

    def _recog_stub(self, B, T, device, dtype):
        return torch.zeros(B, T, self.out_channels, device=device, dtype=dtype)

    def forward(self, x, y_spot_gt=None):
        if x.dim() == 4:
            x = x.squeeze(1)

        stem = self.Stem(x)                                     # (B, dim, T)
        cond = rearrange(stem, 'b c t -> b t c')               # (B, T, dim)

        # Diffusion
        diff_loss, feat, diff_aux = self.diffusion(cond, y_star=y_spot_gt)

        raw_feat = feat.clone()                                 # (B, T, feat_out)

        # Mamba stack
        for mamba_layer in self.mamba_stack:
            feat = mamba_layer(feat) + feat
        feat = self.mamba_norm(feat)

        # Spotting head
        spot_input = self.feat_proj(feat) + self.skip_scale * self.skip_proj(raw_feat)
        spot_prob, spot_logits = self.spot_head(spot_input.transpose(1, 2))

        # Recognition head
        if self.enable_recognition:
            cond_p   = self.cond_proj(cond)
            ring     = torch.cat([feat, cond_p], dim=-1)
            ring     = self.ring_norm(ring)
            attn_out, _ = self.cross_attn(ring, ring, ring)
            ring     = ring + attn_out
            rec_feat = self.recog_head(ring.transpose(1, 2)).transpose(1, 2)
            recog_logits = self.fc(rec_feat)
        else:
            B, T, _ = feat.shape
            recog_logits = self._recog_stub(B, T, feat.device, feat.dtype)

        return spot_prob, recog_logits, diff_loss, diff_aux