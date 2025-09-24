# psam_align.py
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["get_GLA_model", "GLA_model", "PSAMAlign"]


def _pair(v: int | Tuple[int, int]) -> Tuple[int, int]:
    return v if isinstance(v, tuple) else (v, v)


class PatchEmbed(nn.Module):
    """
    PATCH + PROJ (수식 (1))
    - Conv2d(kernel=patch_size, stride=stride)로 패치 토큰화
    - Flatten + LayerNorm으로 토큰 정규화
    """
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: Optional[int] = None) -> None:
        super().__init__()
        patch = _pair(patch_size)
        stride = _pair(patch_size if stride is None else stride)

        self.patch = patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=stride, padding=0, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 패치로 정확히 나누어 떨어지도록 복제 패딩
        H, W = x.shape[-2:]
        ph, pw = self.patch
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x = self.proj(x)              # (B, D, Hp, Wp)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x


class QKAttentionMap(nn.Module):
    """
    linear → query, key  (수식 (2))
    softmax(q k^T) → attention map (수식 (4))
    Value/출력 특징은 사용하지 않음: 맵만 생성.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)
        q = self.q(tokens)                               # (B, N, D)
        k = self.k(tokens)                               # (B, N, D)
        attn = torch.bmm(q, k.transpose(1, 2))           # (B, N, N) = q k^T
        attn = attn / (q.size(-1) ** 0.5)                # scaled
        attn = self.softmax(attn)                        # (B, N, N)
        return attn


class PSAMAlign(nn.Module):
    """
    Patch-based Self-attention Alignment Module (PSAM)
    - 논문 수식 (1)~(5) 충실 구현
    - 교사(Teacher) 경로는 학습 비활성화
    """
    def __init__(self, embed_dim: int = 64, patch_size: int = 8, stride: Optional[int] = None) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.stride = stride if stride is not None else patch_size

        # Lazy build: 첫 forward에서 입력 채널을 보고 모듈 생성
        self.student_proj: Optional[PatchEmbed] = None
        self.teacher_proj: Optional[PatchEmbed] = None
        self.student_qk: Optional[QKAttentionMap] = None
        self.teacher_qk: Optional[QKAttentionMap] = None

    def _build_if_needed(self, c_s: int, c_t: int, device: torch.device) -> None:
        if self.student_proj is not None:
            return
        D, P, S = self.embed_dim, self.patch_size, self.stride

        # PATCH + PROJ (수식 (1))
        self.student_proj = PatchEmbed(c_s, D, P, S).to(device)
        self.teacher_proj = PatchEmbed(c_t, D, P, S).to(device)
        # 교사 경로 동결
        self.teacher_proj.requires_grad_(False)

        # linear → query/key, softmax(q k^T) (수식 (2), (4))
        self.student_qk = QKAttentionMap(D).to(device)
        self.teacher_qk = QKAttentionMap(D).to(device)
        # 교사 경로 동결
        self.teacher_qk.requires_grad_(False)

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            teacher_feat: (B, C_t, H_t, W_t)
            student_feat: (B, C_s, H_s, W_s)
        Returns:
            loss (scalar): MSE(sa_t, sa_s)  (수식 (5))
        """
        device = student_feat.device
        self._build_if_needed(student_feat.shape[1], teacher_feat.shape[1], device)
        assert self.student_proj and self.teacher_proj and self.student_qk and self.teacher_qk

        # 교사 특징 해상도를 학생 특징과 동일하게 맞춤 (일반적 KD 관행)
        teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False)

        # (1) PATCH + PROJ
        s_tokens = self.student_proj(student_feat)   # (B, N, D)
        with torch.no_grad():
            t_tokens = self.teacher_proj(teacher_feat)  # (B, N, D)  (패딩 규칙 동일 → N 일치)

        # (2), (4) Q/K 생성 → softmax(q k^T)
        sa_s = self.student_qk(s_tokens)             # (B, N, N)
        with torch.no_grad():
            sa_t = self.teacher_qk(t_tokens)         # (B, N, N)

        # (5) Loss = MSE(sa_t, sa_s)
        N = sa_s.size(1)
        loss = F.mse_loss(sa_s, sa_t, reduction='sum') / N  # ≈ mean * N
        return loss

    # ---- 옵티마이저에 학생 파라미터만 노출 ----
    def _iter_student_modules(self) -> Iterator[nn.Module]:
        for m in (self.student_proj, self.student_qk):
            if m is not None:
                yield m

    def student_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for m in self._iter_student_modules():
            yield from m.parameters(recurse=recurse)

    # parameters()를 학생 파라미터로 한정
    def parameters(self, recurse: bool = True):  # type: ignore[override]
        yield from self.student_parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        for m_name, m in (("student_proj", self.student_proj), ("student_qk", self.student_qk)):
            if m is None:
                continue
            for pn, p in m.named_parameters(prefix="", recurse=recurse):
                full = f"{m_name}.{pn}" if pn else m_name
                yield full, p

    def get_extra_parameters(self):
        return list(self.student_parameters())


# 원래 코드와 인터페이스 호환을 위한 래퍼
class GLA_model(PSAMAlign):
    def __init__(self, batchsize: Optional[int] = None, **kwargs) -> None:
        del batchsize
        super().__init__(**kwargs)


def get_GLA_model(batchsize: Optional[int] = None, **kwargs) -> GLA_model:
    model = GLA_model(batchsize=batchsize, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
