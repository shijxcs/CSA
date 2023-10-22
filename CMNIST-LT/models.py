import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LeNet(nn.Module):
    def __init__(self, n_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Linear(84, 10)
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet_FE(nn.Module):
    def __init__(self, n_channels=1):
        super(LeNet_FE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        x = self.fc(x)
        return x
    
    @property
    def weight(self):
        return self.fc.weight

class LeNet2_FE(nn.Module):
    def __init__(self, n_channels=1, feat_size=2):
        super(LeNet2_FE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            # nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            # nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, feat_size),
            # nn.BatchNorm1d(feat_size),
            # nn.ReLU(inplace=True),
        )
        self.apply(_weights_init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class NonBiasClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(NonBiasClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_normal_(self.weight)

    def forward(self, x, detach_weights=False):
        if not detach_weights:
            out = F.linear(x, self.weight)
        else:
            out = F.linear(x, self.weight.detach())
        return out

class SphereClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(SphereClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_normal_(self.weight)

    def forward(self, x, detach_weights=False):
        if not detach_weights:
            out = F.linear(F.normalize(x, dim=1), self.weight)
        else:
            out = F.linear(F.normalize(x, dim=1), self.weight.detach())
        return out

class L2NormedClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(L2NormedClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm(2, 0, 1e-5).mul_(1e5)
        
    def forward(self, x, detach_weights=False):
        if not detach_weights:
            out = F.linear(x, F.normalize(self.weight, dim=1))
        else:
            out = F.linear(x, F.normalize(self.weight.detach(), dim=1))
        return out

class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        self.scale = Parameter(torch.Tensor(1))
        self.scale.data.fill_(10.0)

    def forward(self, x, detach_weights=False):
        if not detach_weights:
            out = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
            out *= self.scale
        else:
            out = F.linear(F.normalize(x, dim=1), F.normalize(self.weight.detach(), dim=1))
            out *= self.scale
        return out

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.weight1 = Parameter(torch.Tensor(in_features, in_features))
        self.weight2 = Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_normal_(self.weight1)
        init.kaiming_normal_(self.weight2)

    def forward(self, x):
        out = F.linear(x, self.weight1)
        out = F.relu(out, inplace=True)
        out = F.linear(x, self.weight2)
        return out