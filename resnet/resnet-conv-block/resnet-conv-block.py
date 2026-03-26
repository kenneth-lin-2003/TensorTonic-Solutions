import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        # YOUR CODE HERE
        w1 = self.W1#.transpose(0, 2, 3, 1)
        w2 = self.W2#.transpose(0, 2, 3, 1)
        x1 = relu(x @ w1)
        x2 = relu(x1 @ w2)
        return relu(x2 + x @ self.Ws)
        pass
