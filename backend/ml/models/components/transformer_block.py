import torch.nn as nn

class TrasnformerBlock(nn.Module):
    def __init__(self, *args, num_layers=1, block_id=None):
        super().__init__()
        self.sequence = nn.Sequential(*args)
        self.num_layers = num_layers
        self.block_id = block_id
        
        # 각 레이어에 ID 설정
        for i, layer in enumerate(args):
            if hasattr(layer, "layer_id"):
                layer.layer_id = f"{block_id}_layer_{i}" if block_id else f"layer_{i}"

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.sequence(x)
        return x

    def __repr__(self):
        lines = []
        for i in range(self.num_layers):
            lines.append(f"  # Block 반복 {i+1}/{self.num_layers}")
            for j, layer in enumerate(self.sequence):
                layer_str = repr(layer).replace('\n', '\n    ')
                lines.append(f"    ({j}): {layer_str}")
        return "TransformerBlock(\n" + "\n".join(lines) + "\n)"