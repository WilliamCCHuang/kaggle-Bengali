import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import pretrainedmodels.utils as utils
from efficientnet_pytorch import EfficientNet


class BengaliModel(nn.Module):
    '''
    The pooling:
    Note that the pool here is for GlobalAveragePooling for now
    May change in the future.
    
    In efficientnet:
    The out_channel 32 may be defined in the hyperparameter
    We follow this number for now.
        
    '''
    def __init__(self, model_name, hidden_dim, dropout):
        super(BengaliModel, self).__init__()
        self.model_name = model_name # just for record
        self.hidden_dim = hidden_dim

        if model_name.find('efficientnet') == 0:
            self.cnn = EfficientNet.from_name(model_name)
            self.cnn._conv_stem = nn.Conv2d(1, 32, 3, stride=1, padding=0)
            dim_feats= self.cnn._fc.in_features
        else:
            self.cnn = pretrainedmodels.__dict__[model_name](num_classes=1000, 
                                                              pretrained=None)
            if model_name.find('resnet') == 0:
                self.cnn.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=0)
            elif model_name.find('densenet') == 0:
                self.cnn.features.conv0 = nn.Conv2d(1, 64, 3, stride=1, padding=0)
            elif model_name.find('se_resnet') == 0:
                self.cnn.layer0.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=0)
            dim_feats = self.cnn.last_linear.in_features  
        
        self.linear_x_1 = nn.Linear(dim_feats, hidden_dim)
        self.linear_x_2 = nn.Linear(hidden_dim, 168)
        self.linear_y_1 = nn.Linear(dim_feats, hidden_dim)
        self.linear_y_2 = nn.Linear(hidden_dim, 11)
        self.linear_z_1 = nn.Linear(dim_feats, hidden_dim)
        self.linear_z_2 = nn.Linear(hidden_dim, 7)

        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.pool = nn.AdaptiveAvgPool2d((1,1))
    
    def features(self, input):
        if self.model_name.find('efficientnet') == 0:
            return self.cnn.extract_features(input)
        else:
            return self.cnn.features(input)

    def _logits(self, input, linear_1, linear_2):
        if self.model_name.find('efficientnet') == 0:
            output = self.pool(input)

            if not self.dropout:
                output = self.dropout(output)

            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)

            if not self.dropout:
                output = self.dropout(output)

            output = linear_2(output)

        elif self.model_name.find('resnet') == 0:
            output = self.pool(input)
            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)
            output = self.dropout(output)
            output = linear_2(output)
            
        elif self.model_name.find('densenet') == 0:
            output = F.relu(input, inplace = True)
            output = self.pool(output)
            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)
            output = self.dropout(output)
            output = linear_2(output)
            
        elif self.model_name.find('se_resnet') == 0:
            output = self.pool(input)

            if not self.dropout:
              output = self.dropout(output)

            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)

            if not self.dropout:
              output = self.dropout(output)
            output = linear_2(output)
        
        return output
            
    def logits_x(self, input):
        return self._logits(input, self.linear_x_1, self.linear_x_2)

    def logits_y(self, input):
        return self._logits(input, self.linear_y_1, self.linear_y_2)

    def logits_z(self, input):
        return self._logits(input, self.linear_y_1, self.linear_y_2)

    def forward(self, input):
        output = self.features(input)
        output_x = self.logits_x(output)
        output_y = self.logits_y(output)
        output_z = self.logits_z(output)
        
        return output_x, output_y, output_z


if __name__ == "__main__":
    # produce random square image input
    input = torch.rand((1, 1, 64, 64))

    model = BengaliModel(model_name='se_resnet152', hidden_dim=128, dropout=0.5)
    output_x, output_y, output_z = model(input)
    print(output_x.size())
    print(output_y.size())
    print(output_z.size())
    print(output_x)

