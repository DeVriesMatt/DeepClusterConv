import math
from functools import partial
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from torchsummary import summary
from networks import *


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=True,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3,
                 input_shape=[64, 64, 64, 1],
                 filters=[32, 64, 128, 256, 512],
                 num_features=20,
                 num_clusters=3,
                 bias=True,
                 activations=False):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        self.activations = activations

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_features)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2) * (
                (input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2) * (
                                   (input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2) * filters[4]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        # deemb_lin_features_len = ((input_shape[0] // 2 // 2 - 1) // 2) * (
        #                         (input_shape[1]  // 2 // 2 - 1) // 2) * (
        #                         (input_shape[2]  // 2 // 2 - 1) // 2) * filters[2]

        self.deembedding = nn.Linear(num_features, 512, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2// 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose3d(filters[4], filters[3], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        self.bn5_2 = nn.BatchNorm3d(filters[3])
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose3d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm3d(filters[2])
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm3d(filters[1])
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm3d(filters[0])
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose3d(filters[0], 1, 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        fcdown1 = x
        x = self.fc(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu5_2(x)
        # x = x.view(x.size(0), self.filters[2],
        #            ((self.input_shape[0] // 2 // 2 - 1) // 2),
        #            ((self.input_shape[1] // 2 // 2 - 1) // 2),
        #            ((self.input_shape[2] // 2 // 2 - 1) // 2))
        x = x.view(x.size(0), self.filters[4],
                   ((self.input_shape[0] // 2 // 2 // 2 // 2 //2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 // 2 // 2 // 2 - 1) // 2))
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = nn.Softmax(x)
        return x, clustering_out, extra_out, fcdown1


if __name__ == '__main__':
    model = CAE_5().to('cuda:0')
    print(summary(model, input_size=(1, 64, 64, 64)))
    print(summary(ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes()).to('cuda:0'), input_size=(1, 128, 128, 128)))
