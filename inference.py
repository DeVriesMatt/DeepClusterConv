from networks import CAE_3, CAE_bn3, CAE_bn5, CAE_bn4
from networks_resnet import ResNet, BasicBlock, get_inplanes
from training_functions import *
from datasets import ImageFolder
from torchvision import transforms
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import metrics

model = CAE_bn3(num_features=10, num_clusters=10, input_shape=[28, 28, 28, 1])

# model = ResNet(BasicBlock, layers=[3, 4, 23, 3],
#                block_inplanes=get_inplanes(),
#                input_shape=[64, 64, 64, 1],
#                num_clusters=3,
#                num_features=20)
model.cuda()
model.load_state_dict(torch.load('./nets/CAE_bn3_056.pt'))

model.eval()

data_transforms = transforms.Compose([
            # transforms.Resize(img_size[0:3]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
# Read data from selected folder and apply transformations
sng_cell_path = '/home/mvries/Documents/GitHub/cellAnalysis/SingleCellFull/OPM_Roi_Images_Full_646464_Cluster3/'
pix_path = '/home/mvries/Documents/Datasets/Pix3DVoxels/'
vicky_path = '/home/mvries/Documents/Datasets/VickPlatesStacked/Treatments_plate_002_166464/'
mnist = '/home/mvries/Documents/Datasets/MNIST3D/Train/'

image_dataset = ImageFolder(root=mnist,
                            transform=data_transforms)
# Prepare data for network: schuffle and arrange batches
dataloader = torch.utils.data.DataLoader(image_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4)
features = []
labels = []
for i, data in tqdm(enumerate(dataloader)):
    images, label = data
    labels.append(label.numpy())
    inputs = images.to("cuda:0")
    threshold = 0.0
    inputs = (inputs > threshold).type(torch.FloatTensor).to("cuda:0")
    x, clustering_out, extra_out, fcdown1 = model(inputs)
    features.append(torch.squeeze(extra_out).cpu().detach().numpy())
    # io.imsave('./reconstructed_img/' + str(i).zfill(4) + '.tif',
    #           torch.sigmoid(torch.squeeze(x)).cpu().detach().numpy())

output_array = np.asarray(features)
labels = np.array(labels)
print(labels.shape)
print(output_array.shape)
scalar = StandardScaler()
output_array = scalar.fit_transform(output_array)

# UMAP for vizualization
reducer = umap.UMAP()
embedding = reducer.fit_transform(output_array)
# print(embedding.shape)

# K-means cluster on original data in order to see how UMAP preserves the original clusters
km = KMeans(n_clusters=10, n_init=20)
predictions = km.fit_predict(output_array)

cv_colours = fte_colors = {
     'Coverslip': "#008fd5",
     'NotCoverslip': "#fc4f30",
 }
fte_colors = {
     0: "#008fd5",
     1: "#fc4f30",
     2: 'b',
     3: 'r',
     4: 'g',
     5: 'c',
     6: 'm',
     7: 'y',
     8: 'tan',
     9: 'lime'
 }
# Plot of UMAP with clusters from unreduced data labelled
km_colors = [fte_colors[label] for label in km.labels_]
b = np.zeros((60009, 3))
b[:, 0] = embedding[:, 0]
b[:, 1] = embedding[:, 1]
b[:, 2] = km.labels_  # labels[:, 0]
data = pd.DataFrame(b, columns=['Umap1','Umap2','label'])
facet = sns.lmplot(data=data, x='Umap1', y='Umap2', hue='label',
                   fit_reg=False, legend=True, legend_out=True, scatter_kws={"s": 6})
plt.show()

reduced_pca = PCA(n_components=2).fit_transform(output_array)
b = np.zeros((60009, 3))
b[:, 0] = reduced_pca[:, 0]
b[:, 1] = reduced_pca[:, 1]
b[:, 2] = km.labels_
data = pd.DataFrame(b, columns=['PC1','PC2','label'])
facet_pca = sns.lmplot(data=data, x='PC1', y='PC2', hue='label',
                       fit_reg=False,
                       legend=True,
                       legend_out=True,
                       scatter_kws={"s": 6})
plt.show()


Y = manifold.TSNE(n_components=2, init='pca',
                                 random_state=0).fit_transform(output_array)


b = np.zeros((60009, 3))
b[:, 0] = Y[:, 0]
b[:, 1] = Y[:, 1]
b[:, 2] = km.labels_
data = pd.DataFrame(b, columns=['tsne1','tsne2','label'])
facet_tsne = sns.lmplot(data=data, x='tsne1', y='tsne2', hue='label',
                       fit_reg=False,
                       legend=True,
                       legend_out=True,
                       scatter_kws={"s": 6})
plt.show()

# pd.DataFrame(np.array(km.labels_), columns=['Labels']).to_csv('./OutputFiles/' + 'orig_km_labels.csv')
# centres_df = pd.DataFrame(np.array(km.cluster_centers_))
# centres_df.to_csv('./OutputFiles/' + 'orig_cluster_centers.csv')

closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, output_array)
print(closest)
acc = metrics.metrics.acc(labels, km.labels_)
print('Accuracy: ' + str(acc))
# After 10 clustering epochs, got acc 0.75708
