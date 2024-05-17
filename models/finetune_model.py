from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("finetune_model")
class FinetuneModel(BaseModel):
    def __init__(self):
        super(FinetuneModel, self).__init__()

    def forward(self, inputs, pad_mask):
        if self.frozen_upstream:
            self.model['upstream'].eval()
            with torch.no_grad():
                outputs = self.model['upstream'](inputs, pad_mask, intermediate_rep=True)
        else:
            outputs = self.model['upstream'](inputs, pad_mask, intermediate_rep=True)

        middle = int(outputs.shape[1] / 2)
        # print(f"outputs shape: {outputs.shape}")
        outputs = outputs[:, middle - 5:middle + 5].mean(axis=1)
        out = self.model['prediction_head'](outputs)
        return out
    
    def build_model(self, cfg, upstream_model):
        self.cfg = cfg
        self.upstream = upstream_model
        self.upstream_cfg = self.upstream.cfg
        hidden_dim = self.upstream_cfg.hidden_dim
        self.prediction_head = nn.Sequential(
            # nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2), # Second layer
            # nn.
            nn.Linear(in_features=hidden_dim, out_features=cfg.num_classes)
        )

        for m in self.prediction_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.frozen_upstream = cfg.frozen_upstream

        # make upstream and prediction_head both savable, register them
        self.model = nn.ModuleDict({
            'upstream': self.upstream,
            'prediction_head': self.prediction_head
        })

