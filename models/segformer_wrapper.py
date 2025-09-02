# models/segformer_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import config

_SOURCES = {
    "segformerb0": "nvidia/mit-b0",
    "segformerb1": "nvidia/mit-b1",
    "segformerb3": "nvidia/mit-b3",
    "segformerb5": "nvidia/mit-b5",
}

class SegFormerWrapper(nn.Module):
    def __init__(self, name: str, num_classes: int = config.DATA.NUM_CLASSES):
        super().__init__()
        name = name.lower()
        assert name in _SOURCES, f"Unknown SegFormer name: {name}"
        src = _SOURCES[name]

        cfg = SegformerConfig.from_pretrained(src)
        cfg.num_labels = num_classes
        cfg.id2label = {i: n for i, n in enumerate(config.DATA.CLASS_NAMES)}
        cfg.label2id = {n: i for i, n in enumerate(config.DATA.CLASS_NAMES)}
        cfg.output_hidden_states = True  # 항상 stage feats 계산

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            src, config=cfg, ignore_mismatched_sizes=True
        )

    def forward(self, x, return_feats: bool = False):
        # config에서 이미 output_hidden_states=True 이므로 인자 없이 호출
        out = self.model(pixel_values=x, return_dict=True)
        logits = F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if return_feats:
            feats = getattr(out, "encoder_hidden_states", None)
            if feats is None:
                feats = getattr(out, "hidden_states", None)
            if feats is None or len(feats) < 4:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            feats = tuple(feats[-4:])  # 마지막 4개 stage
            return logits, feats
        return logits
