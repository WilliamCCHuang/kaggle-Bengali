import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pretrainedmodels
import pretrainedmodels.utils as utils
from efficientnet_pytorch import EfficientNet


class BaseCNNModel(nn.Module):
    '''
    The pooling:
    Note that the pool here is for GlobalAveragePooling for now
    May change in the future.
    
    In efficientnet:
    The out_channel 32 may be defined in the hyperparameter
    We follow this number for now.
        
    '''
    def __init__(self, model_name, hidden_dim, dropout):
        super(BaseCNNModel, self).__init__()
        self.model_name = model_name # just for record
        self.hidden_dim = hidden_dim

        # out_channels are different in different models
        if model_name.startswith('efficientnet'):
            self.cnn = EfficientNet.from_name(model_name)
            out_channels = self.cnn._conv_stem.out_channels # get original out_channels
            # print(out_channels)
            self.cnn._conv_stem = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
            dim_feats= self.cnn._fc.in_features
        else:
            self.cnn = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            if model_name.startswith('resnet'):
                out_channels = self.cnn.conv1.out_channels # get original out_channels
                # print(out_channels)
                self.cnn.conv1 = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
            elif model_name.startswith('densenet'):
                out_channels = self.cnn.features.conv0.out_channels # get original out_channels
                # print(out_channels)
                self.cnn.features.conv0 = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
            elif model_name.startswith('se_resnet'):
                out_channels = self.cnn.layer0.conv1.out_channels # get original out_channels
                # print(out_channels)
                self.cnn.layer0.conv1 = nn.Conv2d(1, out_channels, 3, stride=1, padding=0)
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
        if self.model_name.startswith('efficientnet'):
            return self.cnn.extract_features(input)
        else:
            return self.cnn.features(input)

    def _logits(self, input, linear_1, linear_2):
        if self.model_name.startswith('efficientnet'):
            output = self.pool(input)

            if not self.dropout:
                output = self.dropout(output)

            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)

            if not self.dropout:
                output = self.dropout(output)

            output = linear_2(output)

        elif self.model_name.startswith('resnet'):
            output = self.pool(input)
            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)
            output = self.dropout(output)
            output = linear_2(output)
            
        elif self.model_name.startswith('densenet'):
            output = F.relu(input, inplace = True)
            output = self.pool(output)
            output = output.view(output.size(0), -1)
            output = linear_1(output)
            output = F.relu(output)
            output = self.dropout(output)
            output = linear_2(output)
            
        elif self.model_name.startswith('se_resnet'):
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
        return self._logits(input, self.linear_z_1, self.linear_z_2)

    def forward(self, input):
        output = self.features(input)
        output_x = self.logits_x(output)
        output_y = self.logits_y(output)
        output_z = self.logits_z(output)
        
        return output_x, output_y, output_z


if __name__ == "__main__":
    from tqdm import tqdm
    
    # produce random square image input
    input_32x32 = torch.rand((1, 1, 32, 32))
    input_64x64 = torch.rand((1, 1, 64, 64))

    model_names = [
        'resnet18', 'resnet152', # out_channels = 64
        'densenet121', 'densenet161', # out_channels = 64
        'se_resnet50', 'se_resnet152',
        # 'se_resnext50_32x4d', 'se_resnext101_32x4d', # group = 32
        'efficientnet-b0', # out_channels = 32
        'efficientnet-b7' # out_channels = 64
    ]

    for model_name in tqdm(model_names, desc='model'):
        model = BaseCNNModel(model_name=model_name, hidden_dim=128, dropout=0.5)
        model.eval()

        with torch.no_grad():
            try:
                output_x_32x32, output_y_32x32, output_z_32x32 = model(input_32x32)
            except RuntimeError as e:
                print(model_name, '32x32')
                raise ValueError(e)
            try:
                output_x_64x64, output_y_64x64, output_z_64x64 = model(input_64x64)
            except RuntimeError as e:
                print(model_name, '64x64')
        
        logs = 'model name: {}, {}: {}'
        if list(output_x_32x32.size()) != [1, 168]:
            raise RuntimeError(logs.format(model_name, 'output_x_32x32', output_x_32x32.size()))
        if list(output_y_32x32.size()) != [1, 11]:
            raise RuntimeError(logs.format(model_name, 'output_y_32x32', output_y_32x32.size()))
        if list(output_z_32x32.size()) != [1, 7]:
            raise RuntimeError(logs.format(model_name, 'output_z_32x32', output_z_32x32.size()))
        if list(output_x_64x64.size()) != [1, 168]:
            raise RuntimeError(logs.format(model_name, 'output_x_64x64', output_x_64x64.size()))
        if list(output_y_64x64.size()) != [1, 11]:
            raise RuntimeError(logs.format(model_name, 'output_y_64x64', output_y_64x64.size()))
        if list(output_z_64x64.size()) != [1, 7]:
            raise RuntimeError(logs.format(model_name, 'output_z_64x64', output_z_64x64.size()))
