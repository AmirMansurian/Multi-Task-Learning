import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNetBack(nn.Module):
    def __init__(self, input_channels = 1):
        super(ProtoNetBack, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 1),
            conv_block(64, 128),
            conv_block(128, 128),
            conv_block(128, 128),
            conv_block(128, 128),
        )
        
        self.FC1 = nn.Linear(512, 128)
        self.FC2 = nn.Linear(128, 10)


    def get_embedding_size(self, input_size = (3,32,32)):
        device = next(self.parameters()).device
        x = torch.rand([2,*input_size]).to(device)
        with torch.no_grad():
            output = self.forward(x)
            emb_size = output.shape[-1]
        
        del x,output
        torch.cuda.empty_cache()

        return emb_size

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.FC1(x))
        x = self.FC2(x)
        
        return x
        #return self.layers (x).reshape ([x.shape[0] , -1])
    
    