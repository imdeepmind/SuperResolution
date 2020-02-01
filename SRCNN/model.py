import torch.nn as nn

class Model(nn.Modules):
    def __init__(self):
        super(Model, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3,
                            out_channels=64, 
                            kernel_size=9, 
                            padding=4)
        
        self.b1 = nn.BatchNorm2d(64)

        self.c2 = nn.Conv2d(in_channels=64,
                            out_channels=32, 
                            kernel_size=5,
                            padding=2)

        self.b2 = nn.BatchNorm2d(32)

        self.c3 = nn.Conv2d(in_channels=32, 
                            out_channels=3, 
                            kernel_size=5,
                            padding=2)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        a = self.activation(self.b1(self.c1(x)))
        b = self.activation(self.b2(self.c2(a)))
        c = self.c3(b)

        return c

