import torch
import torch.nn as nn
import copy


class CAE_bn3_Seq_2D(nn.Module):
    def __init__(self, input_shape=[64, 64, 64], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=10):
        super(CAE_bn3_Seq_2D, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters

        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        lin_features_len = (((self.input_shape[0] )// 2 // 2 - 1) // 2) * (
                    (self.input_shape[1]// 2 // 2 - 1) // 2) \
                           * filters[2]

        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.encoder = nn.Sequential(nn.Conv2d(self.input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias),
                                     self.relu1_1,
                                     nn.BatchNorm2d(filters[0]),
                                     # nn.MaxPool3d(2),
                                     nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias),
                                     self.relu2_1,
                                     nn.BatchNorm2d(filters[1]),
                                     # nn.MaxPool3d(2),
                                     nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias),
                                     self.relu3_1,
                                     # nn.MaxPool3d(2),
                                     Flatten(),
                                     nn.Linear(lin_features_len, num_features, bias=bias))

    def forward(self, x):
        x = self.encoder(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0] // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 - 1) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        # print(x.shape)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out, x


class EncoderMaxPoolBN3(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=10):
        super(EncoderMaxPoolBN3, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters

        self.conv1 = nn.Conv3d(self.input_shape[3], filters[0], 5, stride=1, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm3d(filters[0])
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.maxpool1 = nn.MaxPool3d(2)
        self.maxpool2 = nn.MaxPool3d(2)
        self.maxpool3 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=1, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm3d(filters[1])
        self.conv3 = nn.Conv3d(filters[1], filters[2], 3, stride=1, padding=0, bias=bias)
        lin_features_len = ((int(self.input_shape[0]) // 2 // 2 - 1) // 2) * (
                (int(self.input_shape[1]) // 2 // 2 - 1) // 2) \
                           * ((int(self.input_shape[2]) // 2 // 2 - 1) // 2) * \
                           filters[2]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.flatten = Flatten()
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)

        self.encoder = nn.Sequential(self.conv1, self.relu1_1, self.bn1_1, self.maxpool1, self.conv2, self.relu2_1,
                                     self.bn2_1, self.maxpool2, self.conv3, self.relu3_1, self.maxpool3, self.flatten,
                                     self.embedding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out # (batch_size, *size)


# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        # represent weight as module parameter
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        #         print('1:  ', x)
        x = x.unsqueeze(1) - self.weight
        #         print('2:  ', x.unsqueeze(1))
        x = torch.mul(x, x)
        #         print('3:  ', x.shape)
        x = torch.sum(x, dim=2)
        #         print('4:  ', x.shape)
        x = 1.0 + (x / self.alpha)
        #         print('5:  ', x.shape)
        x = 1.0 / x
        #         print('6:  ', x.shape)
        x = x ** ((self.alpha + 1.0) / 2.0)
        #         print('7:  ', x.shape)
        x = torch.t(x) / torch.sum(x, dim=1)
        #         print('8:  ', x.shape)
        x = torch.t(x)
        # print('9:  ', x.shape)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


# Convolutional autoencoder directly from DCEC article
class CAE_3(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_3, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
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
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
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
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
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
        return x, clustering_out, extra_out, fcdown1


# Convolutional autoencoder from DCEC article with Batch Norms and Leaky ReLUs
class CAE_bn3(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_bn3, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv3d(self.input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm3d(filters[0])
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm3d(filters[1])
        self.conv3 = nn.Conv3d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        lin_features_len = ((self.input_shape[0] // 2 // 2 - 1) // 2) * (
                    (self.input_shape[1] // 2 // 2 - 1) // 2) \
                           * ((self.input_shape[2] // 2 // 2 - 1) // 2) * \
                           filters[2]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.bn3_2 = nn.BatchNorm3d(filters[1])
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.bn2_2 = nn.BatchNorm3d(filters[0])
        self.deconv1 = nn.ConvTranspose3d(filters[0], input_shape[3], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.size(0), -1)
        fcdown1 = x
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0] // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 - 1) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        # print(x.shape)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out, fcdown1


class CAE_bn3_Seq(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=10):
        super(CAE_bn3_Seq, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters

        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        lin_features_len = (((self.input_shape[0] )// 2 // 2 - 1) // 2) * (
                    (self.input_shape[1]// 2 // 2 - 1) // 2) \
                           * ((self.input_shape[2] // 2 // 2 - 1) // 2) * \
                           filters[2]

        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.bn3_2 = nn.BatchNorm3d(filters[1])
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.bn2_2 = nn.BatchNorm3d(filters[0])
        self.deconv1 = nn.ConvTranspose3d(filters[0], input_shape[3], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.encoder = nn.Sequential(nn.Conv3d(self.input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias),
                                     self.relu1_1,
                                     nn.BatchNorm3d(filters[0]),
                                     # nn.MaxPool3d(2),
                                     nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias),
                                     self.relu2_1,
                                     nn.BatchNorm3d(filters[1]),
                                     # nn.MaxPool3d(2),
                                     nn.Conv3d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias),
                                     self.relu3_1,
                                     # nn.MaxPool3d(2),
                                     Flatten(),
                                     nn.Linear(lin_features_len, num_features, bias=bias))

    def forward(self, x):
        x = self.encoder(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0] // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 - 1) // 2))
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        # print(x.shape)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out, x


# Convolutional autoencoder with 4 convolutional blocks
class CAE_4(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128, 256], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_4, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv3d(input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv3d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv3d(filters[2], filters[3], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 - 1) // 2) * ((input_shape[1] // 2 // 2 // 2 - 1) // 2) \
                           * ((input_shape[2] // 2 // 2 // 2 - 1) // 2) * \
                           filters[3]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose3d(filters[3], filters[2], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose3d(filters[0], input_shape[3], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.size(0), -1)
        fc1down = x
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2))
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out, fc1down


# Convolutional autoencoder with 4 convolutional blocks (BN version)
class CAE_bn4(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128, 256], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_bn4, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv3d(input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm3d(filters[0])
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm3d(filters[1])
        self.conv3 = nn.Conv3d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm3d(filters[2])
        self.conv4 = nn.Conv3d(filters[2], filters[3], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2) * ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2) \
                           * ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2) * \
                           filters[3]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if self.input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose3d(filters[3], filters[2], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm3d(filters[2])
        out_pad = 1 if self.input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm3d(filters[1])
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm3d(filters[0])
        out_pad = 1 if self.input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose3d(filters[0], input_shape[3], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_features, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.size(0), -1)
        fcdown1 = x
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], ((self.input_shape[0] // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 // 2 - 1) // 2))
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
            x = self.tanh(x)
        return x, clustering_out, extra_out, fcdown1


# Convolutional autoencoder with 5 convolutional blocks
class CAE_5(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128, 256, 512], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_5, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv3d(input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv3d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv3d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.conv5 = nn.Conv3d(filters[3], filters[4], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2) * (
                            (input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2) * (
                            (input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2) * filters[4]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv5 = nn.ConvTranspose3d(filters[4], filters[3], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose3d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose3d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose3d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.size(0), -1)
        fcdown1 = x
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[4],
                   ((self.input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2))
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.deconv1(x)
        return x, clustering_out, extra_out, fcdown1


# Convolutional autoencoder with 5 convolutional blocks (BN version)
class CAE_bn5(nn.Module):
    def __init__(self, input_shape=[64, 64, 64, 1], num_clusters=10, filters=[32, 64, 128, 256, 512], leaky=True,
                 neg_slope=0.01, activations=False, bias=True, num_features=100):
        super(CAE_bn5, self).__init__()
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv3d(input_shape[3], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm3d(filters[0])
        self.conv2 = nn.Conv3d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm3d(filters[1])
        self.conv3 = nn.Conv3d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.bn3_1 = nn.BatchNorm3d(filters[2])
        self.conv4 = nn.Conv3d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.bn4_1 = nn.BatchNorm3d(filters[3])
        self.conv5 = nn.Conv3d(filters[3], filters[4], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2) * (
                            (input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2) * (
                            (input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2) * filters[4]
        self.embedding = nn.Linear(lin_features_len, num_features, bias=bias)
        self.deembedding = nn.Linear(num_features, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.size(0), -1)
        fcdown1 = x
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu5_2(x)
        x = x.view(x.size(0), self.filters[4],
                   ((self.input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[1] // 2 // 2 // 2 // 2 - 1) // 2),
                   ((self.input_shape[2] // 2 // 2 // 2 // 2 - 1) // 2))
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