"""
Positional Embedding modules for the multimodal projector.

Ablation map:
  N  "none"       NoSpatialPosEmbed       no global spatial PE; scale embeddings retained
  A  "learned"    LearnedPosEmbed          current baseline (learnable grid)
  B  "linear"     LinearPosEmbed           functional, no Fourier, no log warp
  C  "fourier"    FourierPosEmbed          Fourier, Cartesian (x,y), fixed center
  F  "polar"      PolarPosEmbed            Fourier on polar (r,θ), no log, fixed center
                                            (PoPE-inspired Camp-B baseline)
  D  "log_retina" LogRetinaPosEmbed        log(1+r) warp + Fourier, fixed center
  E  "log_retina" LogRetinaPosEmbed        log(1+r) warp + Fourier, dynamic center
     (controlled by pos_embed_dynamic_center flag)

All classes share the same interface:
    get_pe(h, w, selection_map=None) -> Tensor (1 or B, h*w, hidden_size)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class HighResPosEmbedBase(nn.Module):
    """Abstract base class. All pos embed variants implement get_pe()."""

    def get_pe(
        self,
        h: int,
        w: int,
        selection_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns positional embeddings for a spatial grid of size (h, w).

        Args:
            h, w        : spatial dimensions after projector downsampling
            selection_map: (B, H_orig, W_orig) float mask from PS3.
                          If provided and dynamic_center is enabled,
                          center is computed from the selection centroid.
                          Otherwise ignored.

        Returns:
            Tensor of shape (1, h*w, hidden_size)  if selection_map is None
                         or (B, h*w, hidden_size)  if selection_map is given
                                                    and dynamic_center=True
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ablation N — no global spatial PE (scale embeddings remain controlled)
# ---------------------------------------------------------------------------

class NoSpatialPosEmbed(HighResPosEmbedBase):
    """Return a zero global-position tensor without removing scale embeddings.

    Keeping ``high_res_pos_embed=True`` means the surrounding PS3 code still
    adds the same learned low/high-resolution scale embeddings used by A/C/E/F.
    This isolates the contribution of *spatial* projector PE more cleanly than
    disabling the entire high-resolution positional branch.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        batch = selection_map.shape[0] if selection_map is not None else 1
        return torch.zeros(
            batch,
            h * w,
            self.hidden_size,
            device=self._device_anchor.device,
            dtype=self._device_anchor.dtype,
        )


# ---------------------------------------------------------------------------
# Ablation A — LearnedPosEmbed (current baseline, wrapped for compatibility)
# ---------------------------------------------------------------------------

class LearnedPosEmbed(HighResPosEmbedBase):
    """
    Ablation A: Original learnable grid positional embedding.
    Stores a base (1, base_tokens, C) parameter and resizes with F.interpolate.
    Behaviour is identical to the original high_res_pos_embed parameter.
    """

    def __init__(self, base_token_num: int, hidden_size: int):
        super().__init__()
        # Keep the same parameter name so checkpoint keys are preserved
        # when loading with strict=False.
        self.high_res_pos_embed = nn.Parameter(
            torch.zeros(1, base_token_num, hidden_size)
        )

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        base_h = base_w = int(self.high_res_pos_embed.shape[1] ** 0.5)
        pe = rearrange(self.high_res_pos_embed, "b (h w) c -> b c h w", h=base_h, w=base_w)
        pe = F.interpolate(pe, size=(h, w), mode="area")
        pe = rearrange(pe, "b c h w -> b (h w) c")
        return pe  # (1, h*w, C)


# ---------------------------------------------------------------------------
# Ablation B — LinearPosEmbed
# ---------------------------------------------------------------------------

class LinearPosEmbed(HighResPosEmbedBase):
    """
    Ablation B: Minimal functional encoding.
    Maps normalised (x, y) coordinates directly to hidden_size via a Linear.
    No Fourier expansion, no log warp.  Drastically fewer parameters.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(2, hidden_size)
        # Match C/E/F: all newly introduced PE branches begin as zero so that
        # training starts from the same pretrained-model function.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _make_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Normalised coordinates in [-1, 1]
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)
        coords = torch.stack([grid_x, grid_y], dim=-1)           # (h, w, 2)
        return coords.reshape(1, h * w, 2)                        # (1, h*w, 2)

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        device = self.proj.weight.device
        dtype  = self.proj.weight.dtype
        coords = self._make_grid(h, w, device, dtype)   # (1, h*w, 2)
        return self.proj(coords)                         # (1, h*w, C)


# ---------------------------------------------------------------------------
# Ablation C — FourierPosEmbed
# ---------------------------------------------------------------------------

class FourierPosEmbed(HighResPosEmbedBase):
    """
    Ablation C: Multi-frequency Fourier positional encoding.
    No log warp, fixed image center.
    coords (x, y) ∈ [-1, 1]  →  [sin(2π2^k·x), cos(2π2^k·x),
                                          sin(2π2^k·y), cos(2π2^k·y)]
    → Linear(4*num_freqs → hidden_size)
    """

    def __init__(self, hidden_size: int, num_freqs: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        # NOTE: We do NOT store freqs as a buffer here.
        # 2^N for N>=16 overflows fp16 (max ~65504), causing sin(Inf)=NaN when
        # the projector is cast to fp16.  We recompute in float32 on the fly.
        self.proj = nn.Linear(4 * num_freqs, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (..., 2)  values in [-1, 1]
        returns: (..., 4*num_freqs)
        """
        x = coords[..., 0:1].float()  # float32 for sin/cos accuracy on high freqs
        y = coords[..., 1:2].float()
        # Recompute in float32 every time — avoids fp16 overflow (2^N > 65504 for N>=16)
        f = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float32)
        args_x = 2 * math.pi * x * f
        args_y = 2 * math.pi * y * f
        feats = torch.cat([args_x.sin(), args_x.cos(),
                           args_y.sin(), args_y.cos()], dim=-1)
        return feats.to(dtype=coords.dtype)

    def _make_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)
        return coords.reshape(1, h * w, 2)              # (1, h*w, 2)

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        device = self.proj.weight.device
        dtype  = self.proj.weight.dtype
        coords = self._make_grid(h, w, device, dtype)   # (1, h*w, 2)
        feats  = self._fourier_features(coords)          # (1, h*w, 4K)
        return self.proj(feats)                          # (1, h*w, C)


# ---------------------------------------------------------------------------
# Ablation F — PolarPosEmbed  (PoPE-inspired Camp-B baseline)
# ---------------------------------------------------------------------------

class PolarPosEmbed(HighResPosEmbedBase):
    """
    Ablation F: Polar-coordinate Fourier PE (PoPE-inspired, Camp B).

    Uses (r, θ) polar coordinates WITHOUT log compression:
        r = sqrt(x²+y²) / sqrt(2)   ∈ [0, 1]
        θ = atan2(y, x) / π         ∈ [-1, 1]

    Fourier features are then applied to (r, θ) and projected linearly.

    Key differences from LogRetinaPosEmbed:
      - No log compression (uniform radial mapping)
      - No dynamic centroid (fixed origin at image center)
      - Fourier features operate directly on (r, theta); LogRetina converts
        the warped radius back to (x', y') before applying Fourier features

    Consequently, F is a practical uniform-polar baseline, not a one-variable
    isolation of the logarithmic warp and not a reproduction of sequence-level
    PoPE.
    """

    def __init__(self, hidden_size: int, num_freqs: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Linear(4 * num_freqs, hidden_size)
        # Zero-init: consistent with C and E
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 2) → (..., 4*num_freqs)"""
        x = coords[..., 0:1].float()  # float32 to avoid fp16 overflow at high freqs
        y = coords[..., 1:2].float()
        f = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float32)
        args_x = 2 * math.pi * x * f
        args_y = 2 * math.pi * y * f
        feats = torch.cat([args_x.sin(), args_x.cos(),
                           args_y.sin(), args_y.cos()], dim=-1)
        return feats.to(dtype=coords.dtype)

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        device = self.proj.weight.device
        dtype  = self.proj.weight.dtype
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)

        # Polar coordinates (fixed center at origin)
        r     = torch.sqrt(grid_x ** 2 + grid_y ** 2).clamp(min=1e-8)
        r_norm = (r / math.sqrt(2)).clamp(0.0, 1.0)   # normalise to [0, 1]
        theta  = torch.atan2(grid_y, grid_x) / math.pi  # normalise to [-1, 1]

        coords = torch.stack([r_norm, theta], dim=-1)  # (h, w, 2)
        coords = coords.reshape(1, h * w, 2)           # (1, h*w, 2)
        feats  = self._fourier_features(coords)         # (1, h*w, 4K)
        return self.proj(feats)                         # (1, h*w, C)


# ---------------------------------------------------------------------------
# Ablation D & E — LogRetinaPosEmbed  (main innovation)
# ---------------------------------------------------------------------------

class LogRetinaPosEmbed(HighResPosEmbedBase):
    """
    Ablation D (dynamic_center=False) / E (dynamic_center=True).

    Biologically-inspired log-polar positional encoding:
      1. Compute radial distance r from center
      2. Compress r with log(1 + alpha * r) / log(1 + alpha)   [fovea effect]
      3. Preserve angle theta
      4. Convert back to warped cartesian (x', y')
      5. Fourier features on (x', y')
      6. Linear(4*num_freqs → hidden_size)

    Low-res  path: center = image center (fixed)
    High-res path: center = centroid of PS3 selection_map (dynamic, Ablation E)
                   center = image center (fixed,           Ablation D)

    Both paths share the same proj layer.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha: float = 5.0,
        num_freqs: int = 32,
        dynamic_center: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.num_freqs = num_freqs
        self.dynamic_center = dynamic_center

        # NOTE: We do NOT store freqs as a buffer here.
        # 2^N for N>=16 overflows fp16 (max ~65504), causing sin(Inf)=NaN when
        # the projector is cast to fp16.  We recompute in float32 on the fly.

        # Shared projection for both low-res and high-res paths
        self.proj = nn.Linear(4 * num_freqs, hidden_size)
        # Zero-init: PE starts at 0 so training begins at the pretrained baseline.
        # This prevents loss explosion from an out-of-distribution random PE.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_warp(self, r: torch.Tensor) -> torch.Tensor:
        """r → log(1 + alpha*r) / log(1 + alpha), maps [0,1] → [0,1]."""
        return torch.log1p(self.alpha * r) / math.log(1.0 + self.alpha)

    def _fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 2) → (..., 4*num_freqs)"""
        x = coords[..., 0:1].float()  # float32 for sin/cos accuracy on high freqs
        y = coords[..., 1:2].float()
        # Recompute in float32 every time — avoids fp16 overflow (2^N > 65504 for N>=16)
        f = 2.0 ** torch.arange(self.num_freqs, device=x.device, dtype=torch.float32)
        args_x = 2 * math.pi * x * f
        args_y = 2 * math.pi * y * f
        feats = torch.cat([args_x.sin(), args_x.cos(),
                           args_y.sin(), args_y.cos()], dim=-1)
        return feats.to(dtype=coords.dtype)

    def _make_warped_coords(
        self,
        h: int,
        w: int,
        cx: float,
        cy: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build log-retina warped coordinates for an (h, w) grid.

        cx, cy : center in normalised [-1, 1] space

        Returns: (1, h*w, 2)
        """
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)

        dx = grid_x - cx  # offset from center
        dy = grid_y - cy

        r     = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1e-8)
        theta = torch.atan2(dy, dx)

        # Normalise r to [0, 1] using the max possible distance in [-1,1]²
        r_norm  = r / (2.0 ** 0.5)           # max dist in [-1,1]² is sqrt(2)
        r_norm  = r_norm.clamp(0.0, 1.0)
        r_warp  = self._log_warp(r_norm)      # [0, 1] → [0, 1] compressed

        x_warp = r_warp * torch.cos(theta)
        y_warp = r_warp * torch.sin(theta)

        coords = torch.stack([x_warp, y_warp], dim=-1)  # (h, w, 2)
        return coords.reshape(1, h * w, 2)               # (1, h*w, 2)

    @staticmethod
    def _selection_centroid(selection_map: torch.Tensor) -> tuple:
        """
        Compute the (cx, cy) centroid of a selection_map in normalised [-1,1].

        selection_map: (B, H, W) float, values in [0, 1]
        Returns: cx (B,), cy (B,)  tensors in [-1, 1]
        """
        B, H, W = selection_map.shape
        device   = selection_map.device
        dtype    = selection_map.dtype

        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        mass = selection_map.sum(dim=(1, 2)).clamp(min=1e-8)  # (B,)
        cx = (selection_map * grid_x.unsqueeze(0)).sum(dim=(1, 2)) / mass  # (B,)
        cy = (selection_map * grid_y.unsqueeze(0)).sum(dim=(1, 2)) / mass  # (B,)
        return cx, cy  # each (B,)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_pe(self, h: int, w: int, selection_map=None) -> torch.Tensor:
        device = self.proj.weight.device
        dtype  = self.proj.weight.dtype

        # ---- Dynamic center (Ablation E, high-res path) ----
        if self.dynamic_center and selection_map is not None:
            cx, cy = self._selection_centroid(
                selection_map.to(device=device, dtype=dtype)
            )  # each (B,)
            B = cx.shape[0]

            # Build per-sample warped coords and stack
            pe_list = []
            for b in range(B):
                coords = self._make_warped_coords(
                    h, w,
                    cx[b].item(), cy[b].item(),
                    device, dtype
                )                                      # (1, h*w, 2)
                feats = self._fourier_features(coords) # (1, h*w, 4K)
                pe_list.append(self.proj(feats))       # (1, h*w, C)
            return torch.cat(pe_list, dim=0)           # (B, h*w, C)

        # ---- Fixed center (Ablation D, or low-res path) ----
        coords = self._make_warped_coords(h, w, 0.0, 0.0, device, dtype)  # (1, h*w, 2)
        feats  = self._fourier_features(coords)                            # (1, h*w, 4K)
        return self.proj(feats)                                            # (1, h*w, C)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pos_embed(config) -> HighResPosEmbedBase:
    """
    Factory function.  Reads config fields:
        config.pos_embed_type          : str   "none" | "learned" | "linear" |
                                                "fourier" | "polar" | "log_retina"
        config.mm_low_res_token_num    : int   (only for "learned")
        config.hidden_size             : int
        config.pos_embed_alpha         : float  default 5.0
        config.pos_embed_num_freqs     : int    default 32
        config.pos_embed_dynamic_center: bool   default True

    Ablation mapping:
        "none"       → N  NoSpatialPosEmbed       (zero global spatial PE)
        "learned"    → A  LearnedPosEmbed
        "linear"     → B  LinearPosEmbed
        "fourier"    → C  FourierPosEmbed          (Cartesian Fourier)
        "polar"      → F  PolarPosEmbed             (polar Fourier, no log)
        "log_retina" → D  LogRetinaPosEmbed(dynamic_center=False)
                     → E  LogRetinaPosEmbed(dynamic_center=True)
    """
    embed_type      = getattr(config, "pos_embed_type", "learned")
    hidden_size     = config.hidden_size
    alpha           = getattr(config, "pos_embed_alpha", 5.0)
    num_freqs       = getattr(config, "pos_embed_num_freqs", 32)
    dynamic_center  = getattr(config, "pos_embed_dynamic_center", True)

    if embed_type == "none":
        return NoSpatialPosEmbed(hidden_size)

    elif embed_type == "learned":
        base_token_num = config.mm_low_res_token_num
        return LearnedPosEmbed(base_token_num, hidden_size)

    elif embed_type == "linear":
        return LinearPosEmbed(hidden_size)

    elif embed_type == "fourier":
        return FourierPosEmbed(hidden_size, num_freqs)

    elif embed_type == "polar":
        return PolarPosEmbed(hidden_size, num_freqs)

    elif embed_type == "log_retina":
        return LogRetinaPosEmbed(hidden_size, alpha, num_freqs, dynamic_center)

    else:
        raise ValueError(
            f"Unknown pos_embed_type: '{embed_type}'. "
            f"Choose from: 'none', 'learned', 'linear', 'fourier', "
            f"'polar', 'log_retina'."
        )
