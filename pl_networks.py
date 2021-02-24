from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import copy
from torchvision import transforms

from networks import ClusterlingLayer
from loss_functions import *
from datasets import ImageFolder


class Cae3Encoder(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True):
        super(Cae3Encoder, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        #         print(input_shape)
        self.filters = filters
        self.conv1 = nn.Conv3d(1, filters[0], 5, stride=2, padding=2, bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv3d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        #         lin_features_len = 18432
        lin_features_len = 43904  # 946176 #3813248
        # ((input_shape[0]//2//2-1) // 2) * ((input_shape[1]//2//2-1) // 2) * ((input_shape[2]//2//2-1) // 2) * filters[2]
        #         print(lin_features_len)
        #         self.fc_down1 = nn.Linear(lin_features_len, 1000)
        #         self.fc_down2 = nn.Linear(10000, 500)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        # self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        # #         self.fc_up1 = nn.Linear(500, 10000)
        # #         self.fc_up2 = nn.Linear(1000, lin_features_len)
        # out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        # self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 3, stride=2, output_padding=out_pad, padding=0,
        #                                   bias=bias)
        # out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        # self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, output_padding=out_pad, padding=2,
        #                                   bias=bias)
        # out_pad = 1 if input_shape[0] % 2 == 0 else 0
        # self.deconv1 = nn.ConvTranspose3d(filters[0], 1, 5, stride=2, output_padding=out_pad, padding=2, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        # self.relu1_2 = copy.deepcopy(self.relu)
        # self.relu2_2 = copy.deepcopy(self.relu)
        # self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        # print(x.shape)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        #         print(x.shape)

        #         x = self.fc_down1(x)
        fcdown1 = x
        #         x = self.fc_down2(x)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        # x = self.deembedding(x)
        # #         x = self.fc_up1(x)
        # #         x = self.fc_up2(x)
        # x = self.relu1_2(x)
        # x = x.view(x.size(0), 128, 7, 7, 7)
        # #  self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        # x = self.deconv3(x)
        # x = self.relu2_2(x)
        # x = self.deconv2(x)
        # x = self.relu3_2(x)
        # x = self.deconv1(x)
        # # print(x.shape)
        # if self.activations:
        #     x = self.tanh(x)
        return x, clustering_out, extra_out, fcdown1


class Cae3Decoder(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True):
        super(Cae3Decoder, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        #         print(input_shape)
        self.filters = filters
        # self.conv1 = nn.Conv3d(1, filters[0], 5, stride=2, padding=2, bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        # self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        # self.conv3 = nn.Conv3d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        #         lin_features_len = 18432
        lin_features_len = 43904  # 946176 #3813248
        # ((input_shape[0]//2//2-1) // 2) * ((input_shape[1]//2//2-1) // 2) * ((input_shape[2]//2//2-1) // 2) * filters[2]
        #         print(lin_features_len)
        #         self.fc_down1 = nn.Linear(lin_features_len, 1000)
        #         self.fc_down2 = nn.Linear(10000, 500)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        #         self.fc_up1 = nn.Linear(500, 10000)
        #         self.fc_up2 = nn.Linear(1000, lin_features_len)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 3, stride=2, output_padding=out_pad, padding=0,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, output_padding=out_pad, padding=2,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose3d(filters[0], 1, 5, stride=2, output_padding=out_pad, padding=2, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        # self.relu1_1 = copy.deepcopy(self.relu)
        # self.relu2_1 = copy.deepcopy(self.relu)
        # self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu1_1(x)
        # x = self.conv2(x)
        # x = self.relu2_1(x)
        # x = self.conv3(x)
        # # print(x.shape)
        # if self.activations:
        #     x = self.sig(x)
        # else:
        #     x = self.relu3_1(x)
        # x = x.view(x.size(0), -1)
        # #         print(x.shape)
        #
        # #         x = self.fc_down1(x)
        # fcdown1 = x
        # #         x = self.fc_down2(x)
        # x = self.embedding(x)
        # extra_out = x
        # clustering_out = self.clustering(x)
        x = self.deembedding(x)
        #         x = self.fc_up1(x)
        #         x = self.fc_up2(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), 128, 7, 7, 7)
        #  self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.deconv1(x)
        # print(x.shape)
        if self.activations:
            x = self.tanh(x)
        return x


class Lit_CAE_3(pl.LightningModule):
    def __init__(self, params, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, encoder='CAE_3', loss=FocalTverskyLoss()):
        super(Lit_CAE_3, self).__init__()
        self.num_clusters = num_clusters
        self.activations = activations
        self.bias = bias
        self.pretrained = False
        self.input_shape = input_shape
        self.filters = filters
        self.neg_slope = neg_slope
        self.leaky = leaky
        self.loss = loss

        if encoder == 'CAE_3':
            self.encoder = Cae3Encoder(self.input_shape, self.num_clusters, self.filters, self.leaky,
                                       self.neg_slope, self.activations, self.bias)
            self.decoder = Cae3Decoder(self.input_shape, self.num_clusters, self.filters, self.leaky,
                                       self.neg_slope, self.activations, self.bias)

        # Unpack parameters
        self.txt_file = params['txt_file']
        self.pretrained = params['model_files'][1]
        self.pretrain = params['pretrain']
        self.print_freq = params['print_freq']
        # self.dataset_size = params['dataset_size']
        # self.device = params['device']
        self.batch = params['batch']
        self.batch_size = params['batch']
        self.pretrain_epochs = params['pretrain_epochs']
        self.gamma = params['gamma']
        self.update_interval = params['update_interval']
        self.tol = params['tol']
        self.weight_pretrain = params['weight_pretrain']
        self.rate_pretrain = params['rate_pretrain']

    def forward(self, x):
        embedding, clustering_out, extra_out, fcdown1 = self.encoder(x)
        return embedding, clustering_out, fcdown1

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        embedding, clustering_out, extra_out, fcdown1 = self.encoder(inputs)
        outputs = self.decoder(embedding)
        loss = self.loss(outputs, inputs)
        self.log('Train loss', loss, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.rate_pretrain,
                                     weight_decay=self.weight_pretrain)
        return optimizer


class LitSingleCellData(pl.LightningDataModule):
    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def __init__(self, params):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.params = params
        self.path = params['output_dir']
        self.data_dir = params['data_dir']
        self.batch_size = self.params['batch']
        self.num_workers = self.params['workers']

    def prepare_data(self, *args, **kwargs):
        self.train_dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        self.test_dataset = ImageFolder(root=self.data_dir, transform=self.transform)

    def setup(self, stage=None):
        if stage =='train' or stage is None:
            self.dataset = self.train_dataset
        else:
            self.dataset = self.test_dataset

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
