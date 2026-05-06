import torch.nn as nn

class ModelWithHead(nn.Module):
    """
    Generic wrapper to add a trainable projection head to a frozen backbone.
    """
    def __init__(self, base_model, input_dim, head_dim=512, output_dim=None):
        super(ModelWithHead, self).__init__()
        self.base_model = base_model
        if output_dim is None:
            output_dim = input_dim
            
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.base_model(x)
        return self.projection_head(features)
