import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        # Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        
        self.conv_1 = nn.Conv1d(k,64, 1)
        self.bn_conv_1 = nn.BatchNorm1d(64)

        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.bn_conv_2 = nn.BatchNorm1d(128)

        self.conv_3 = nn.Conv1d(128, 1024, 1)
        self.bn_conv_3 = nn.BatchNorm1d(1024)
        
        # Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU

        self.linear_1 = nn.Linear(1024, 512)
        self.bn_lin_1 = nn.BatchNorm1d(512)

        self.linear_2 = nn.Linear(512, 256)
        self.bn_lin_2 = nn.BatchNorm1d(256)

        self.linear_3 = nn.Linear(256, k**2)

        self.identity = torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2)
        self.k = k

    def forward(self, x):# x shape: (batch_size, 3->channels, num_points:1024)
        b = x.shape[0]

        # Pass input through layers, applying the same max operation as in PointNetEncoder
        # No batch norm and relu after the last Linear layer
    
        x = F.relu(self.bn_conv_1(self.conv_1(x)))
        x = F.relu(self.bn_conv_2(self.conv_2(x)))
        x = F.relu(self.bn_conv_3(self.conv_3(x))) 

        # Max Pool operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_lin_1(self.linear_1(x)))
        x = F.relu(self.bn_lin_2(self.linear_2(x)))
        x = self.linear_3(x)

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1).to(x.device)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super().__init__()

        # Define convolution layers, batch norm layers, and ReLU
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        # First layer: 3->64
        x = F.relu(self.bn1(self.conv1(x)))

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # Layers 2 and 3: 64->128, 128->1024
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        # Define convolutions, batch norms, and ReLU
        self.conv0 = nn.Conv1d(1088, 512, 1)
        self.bn0 = nn.BatchNorm1d(512)

        self.conv1 = nn.Conv1d(512, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, self.num_classes, 1)


    def forward(self, x):
        x = self.encoder(x)
        # Pass x through all layers, no batch norm or ReLU after the last conv layer
        
        
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        x = x.transpose(2, 1).contiguous()
        return x
