from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultHead(nn.Module):

    def __init__(self, d_model, n_classes, mode):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.linear = nn.Linear(d_model, n_classes) 
    
    def forward(self, x):
        return self.linear(x)



class CustomWrapper(nn.Module):

    def __init__(self, backbone, head=None, n_times=8, drop_rate=0.4, dropout=None):
        super().__init__()

        self.backbone = backbone
        self.n_times = n_times
        self.drop_rate = drop_rateP

        if dropout is not None:
            self.dropout = dropout
        else: 
            self.dropout = nn.Dropout(self.drop_rate)
        
        if head is not None:
            self.head = head 
        else:
            self.head = DefaultHead(self.drop_rate)

    
    def forward(self, x):
        ms_output = deque(maxlen=self.n_times)
        x = self.backbone(x)
        for _ in range(self.n_times):
            x_dropped = self.dropout(x)
            x_dropped = self.head(x_dropped)
            ms_output.append(x_dropped)
        
        output = torch.mean(ms_output)
        return output



class MultisampleWrapper(nn.Module):

    """ Wrapper for any backbone from original papper
        with fixed head
        BACKBONE OUTPUT SHOULD HAVE 3 DIMENSIONS (CxHxW)

        Parameters
        ----------
            backbone: nn.Module
                Base network for wrapper 
            
            n_classes: int 
                Number of clsases of output dimension
            
            n_time: int 
                Number of sampled heads 
            
            dropout: None, list, tuple:
                If None defaul dropout with 40% and 30% ratio are used
                If list or tupele you can use custom dropout with custom ratios
    """

    def __init__(self, backbone, n_classes, n_times=8, dropout=None):
        super().__init__()
        self.backbone = backbone
        self.n_times = n_times
        self.drop_rate = drop_rateP
        self.n_classes = n_classes
        if dropout is not None:
            self.dropout40 = dropout[0]
            self.dropout30 = dropout[1]
        else: 
            self.dropout40 = nn.Dropout(0.4)
            self.dropout30 = nn.Dropout(0.3)

        self.bn = nn.BatchNorm2d(192)
        self.fc = nn.Linear(192, 512)
        self.classifier = nn.Linear(512, n_classes)

    
    def forward(self, x):
        ms_output = deque(maxlen=self.n_times)
        x = self.backbone(x)
        for i in range(self.n_times//2):
            if (i & 1) != 0:
                x_drop = x.flip(3)
            else:
                x_drop = x 
            x_drop = F.max_pool2d(x_drop)
            x_drop = self.bn(x_drop)
            for j in range(0, 2):
                x_drop = self.dropout40(x_drop)
                if j == 1:
                    x_drop = x_drop.flip(3)
                    x_drop = F.relu_(self.fc(x_drop))
                    x_drop = self.dropout30(x_drop)
                    x_drop = self.classifier(x_drop)
                    ms_output.append(x_drop)
        output = torch.mean(ms_output, axis=1)
        return output

