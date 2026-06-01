import torch.nn as nn


class ModelWithHead(nn.Module):
    """
    Trainable projection head on top of a frozen backbone.

    `train()` is overridden so the backbone always stays in `eval()` mode —
    keeps BatchNorm/Dropout layers in inference mode while head fine-tunes.
    """

    def __init__(self, base_model, input_dim, head_dim=512, output_dim=None):
        super().__init__()
        self.base_model = base_model
        if output_dim is None:
            output_dim = input_dim
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, output_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x):
        features = self.base_model(x)
        return self.projection_head(features)
