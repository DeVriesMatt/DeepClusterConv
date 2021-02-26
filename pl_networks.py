from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import copy
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np

from networks import ClusterlingLayer
from loss_functions import *
from datasets import ImageFolder
from metrics import metrics


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
                 neg_slope=0.01, activations=False, bias=True, encoder='CAE_3', loss_rec=FocalTverskyLoss(),
                 loss_clus=nn.KLDivLoss(size_average=False)):
        super(Lit_CAE_3, self).__init__()
        self.num_clusters = num_clusters
        self.activations = activations
        self.bias = bias
        self.pretrained = False
        self.input_shape = input_shape
        self.filters = filters
        self.neg_slope = neg_slope
        self.leaky = leaky
        self.loss_rec = loss_rec
        self.loss_clus = loss_clus
        self.params = params

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
        self.update_interval = params['update_interval']

    def forward(self, x):
        embedding, clustering_out, extra_out, fcdown1 = self.encoder(x)
        output = self.decoder(embedding)
        return embedding, clustering_out, fcdown1, output

    def on_train_start(self) -> None:
        pass
        # kmeans(self, copy.deepcopy(self.params['dataloader']), self.params)
        # print('\nUpdating target distribution')
        # self.output_distribution, self.labels, self.preds_prev = calculate_predictions(self,
        #                                                                 copy.deepcopy(self.params['dataloader']),
        #                                                                 self.params)
        # self.target_distribution = target(self.output_distribution)
        # self.nmi = metrics.nmi(self.labels, self.preds_prev)
        # self.ari = metrics.ari(self.labels, self.preds_prev)
        # self.acc = metrics.acc(self.labels, self.preds_prev)
        # log_dict = {'NMI': nmi,
        #             'ARI': ari,
        #             'Acc': acc,
        #             'Reconstruction loss': nmi,
        #             'Clustering loss': nmi,
        #             'Total loss': nmi}
        # self.logger.log_metrics(log_dict)

    def on_epoch_end(self) -> None:
        pass
        # self.output_distribution, self.labels, self.preds = calculate_predictions(self,
        #                                                                           self.params['dataloader'],
        #                                                                           self.params)
        # self.target_distribution = target(self.output_distribution)
        # self.nmi = metrics.nmi(self.labels, self.preds)
        # self.ari = metrics.ari(self.labels, self.preds)
        # self.acc = metrics.acc(self.labels, self.preds)

    def training_step(self, batch, batch_idx):
        # print(self.current_epoch)
        inputs, labels = batch
        threshold = 0.0
        inputs = (inputs > threshold).type_as(inputs)
        embedding, clustering_out, extra_out, fcdown1 = self.encoder(inputs)
        outputs = self.decoder(embedding)
        loss = self.loss_rec(outputs, inputs)
        self.logger.log_graph(self, inputs)
        self.logger.log_metrics({'Loss': loss.item()})
        return loss

        # if self.params['mode'] == 'train_full':
        #     inputs, labels = batch
        #     threshold = 0.0
        #     inputs = (inputs > threshold).type_as(inputs)
        #     batch_size = self.batch_size
        #
        #     # Uptade target distribution, chack and print performance
        #     self.tar_dist = self.target_distribution[((batch_idx - 1) * batch_size):(batch_idx * batch_size), :]
        #     self.tar_dist = torch.from_numpy(self.tar_dist).type_as(inputs)
        #
        #     embedding, clustering_out, extra_out, fcdown1 = self.encoder(inputs)
        #     outputs = self.decoder(embedding)
        #     comb_loss = CombinedLoss()
        #     loss, rec_loss, clus_loss = comb_loss(outputs,
        #                                           inputs,
        #                                           self.tar_dist,
        #                                           clustering_out,
        #                                           self.gamma,
        #                                           batch_size)
        #     # log_dict = {'NMI': self.nmi,
        #     #             'ARI': self.ari,
        #     #             'Acc': self.acc,
        #     #             'Reconstruction loss': rec_loss,
        #     #             'Clustering loss': clus_loss,
        #     #             'Total loss': loss}
        #     # self.logger.log_metrics(log_dict)
        #     self.log('Train loss', loss, prog_bar=True, on_step=True)
        #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.rate_pretrain,
                                     weight_decay=self.weight_pretrain)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer,
                                                                     step_size=self.params['sched_step_pretrain'],
                                                                     gamma=self.params['sched_gamma_pretrain']),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]


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


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(params['device'])
        outputs, clustering_out, fcdown1, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.encoder.clustering.set_weight(weights.to(params['device']))
    # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])
        embedding, outputs, fcdown1, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist
