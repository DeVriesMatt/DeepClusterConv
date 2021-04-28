# import packages
import os
import time
import torch
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.Image
import torchvision

from sklearn import manifold
from sklearn.cluster import KMeans
import io as inout
import torch.nn.functional as F

from sklearn.decomposition import PCA


# module visualizations.py
from datetime import datetime
# import visdom
# from visdom import Visdom
from skimage import io

# import from files
import loss_functions
from metrics import print_both, tensor2img, metrics
from utils import create_dir_if_not_exist

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# class Visualizations:
#     def __init__(self, env_name='main'):
#         if env_name is None:
#             env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
#         self.env_name = env_name
#         self.vis = visdom.Visdom(env=self.env_name)
#         self.loss_win = None
#
#     def plot_loss(self, loss, step):
#         self.loss_win = self.vis.line(
#             [loss],
#             [step],
#             win=self.loss_win,
#             update='append' if self.loss_win else None,
#             opts=dict(
#                 xlabel='Step',
#                 ylabel='Loss',
#                 title='Loss (mean per 10 steps)',
#             )
#         )
#
#
# class VisdomLinePlotter(object):
#     """Plots to Visdom"""
#
#     def __init__(self, env_name='main'):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}
#
#     def plot(self, var_name, split_name, title_name, x, y):
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
#                 legend=[split_name],
#                 title=title_name,
#                 xlabel='Epochs',
#                 ylabel=var_name
#             ))
#         else:
#             self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
#                           update='append')


# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params, dataloader_inference):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    #     './SingleCellFull/' + 'CAE3_orig_paddedWithin_64128128_3Clusters_need to continueTraining' + '.pt'
    #
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    print(update_interval)
    tol = params['tol']
    tsne_epochs = params['tsne_epochs']
    output_dir = params['output_dir']
    name = params['name']
    q_power = params['q_power']
    rot_loss_weight = params['rot_loss_weight']

    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1],
                                           pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0000002, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.00000000001, 0.1)
    else:
        try:
            print(pretrained)
            model.load_state_dict(torch.load(pretrained)['model_state_dict'])
            trained_model = copy.deepcopy(model)
            model = trained_model
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0000002, momentum=0.9)
            # optimizer.load_state_dict(torch.load(pretrained)['optimizer_state_dict'])
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.00000000001, 0.1)
            print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0000002, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.00000000001, 0.1)

        # Initialize the visualization environment
    # vis = Visualizations()
    # global plotter
    # plotter = VisdomLinePlotter()
    # TODO: seems to be problem with model not training when loading from pretrained weights


    # Initialise clusters
    print_both(txt_file, '\nInitializing cluster centers based on K-means')
    km, reduced_pca = kmeans(model, copy.deepcopy(dl), params)
    b = np.zeros((11021, 3))
    b[:, 0] = reduced_pca[:, 0]
    b[:, 1] = reduced_pca[:, 1]
    b[:, 2] = km.labels_ # km.labels_
    print(km.labels_)
    data = pd.DataFrame(b, columns=['PC1', 'PC2', 'label'])
    facet_pca = sns.lmplot(data=data, x='PC1', y='PC2', hue='label',
                           fit_reg=False,
                           legend=True,
                           legend_out=True,
                           scatter_kws={"s": 6})
    plt.show()

    print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    print_both(txt_file, '\nUpdating target distribution')
    output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params)
    target_distribution = target(output_distribution)
    nmi = metrics.nmi(labels, preds_prev)
    ari = metrics.ari(labels, preds_prev)
    acc = metrics.acc(labels, preds_prev)
    print_both(txt_file,
               'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

    if board:
        niter = 0
        writer.add_scalar('/NMI', nmi, niter)
        writer.add_scalar('/ARI', ari, niter)
        writer.add_scalar('/Acc', acc, niter)

    update_iter = 1
    finished = False

    # Go through all epochs
    for epoch in range(num_epochs):

        print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        # TODO: check if the optimizer is not set correctly
        print_both(txt_file, "Learning Rate: {}".format(optimizer.param_groups[0]['lr']))
        # print_both(txt_file, "Learning Rate: {}".format(optimizers[0].param_groups[0]['lr']))
        print_both(txt_file, '-' * 10)

        # Performs t-sne on extracted features
        if (epoch + 1) % tsne_epochs == 0:
            print_both(txt_file, 'Performing t-SNE on extracted features')
            model.eval()
            features = []
            labs = []
            for data in dataloader_inference:
                images, label, _ = data
                labs.append(label.cpu().detach().numpy())
                inputs = images.to("cuda:0")
                threshold = 0.0
                inputs = (inputs > threshold).type(torch.FloatTensor).to("cuda:0")
                x, clustering_out, extra_out, fcdown1 = model(inputs)
                # print(clustering_out)
                features.append(torch.squeeze(extra_out).cpu().detach().numpy())

            output_array = np.asarray(features)
            scalar = StandardScaler()
            output_array = scalar.fit_transform(output_array)

            labs = np.array(labs)

            Y = manifold.TSNE(n_components=2, init='pca',
                              random_state=0).fit_transform(output_array)

            b = np.zeros((len(features), 3))
            b[:, 0] = Y[:, 0]
            b[:, 1] = Y[:, 1]
            b[:, 2] = km.labels_  #labs[:,0] #
            tsne_data = pd.DataFrame(b, columns=['tsne1', 'tsne2', 'label'])
            facet_tsne = sns.lmplot(data=tsne_data, x='tsne1', y='tsne2', hue='label',
                                    fit_reg=False,
                                    legend=True,
                                    legend_out=True,
                                    scatter_kws={"s": 6})
            facet_tsne.set_titles('t-SNE of extracted features at epoch {}'.format(epoch+1))
            save_dir = output_dir + 't_sne/' + name + '/'
            save_path = save_dir + "t_sne_epoch{}.png".format(epoch + 1)
            create_dir_if_not_exist(save_dir)
            facet_tsne.savefig(save_path)

            # Turn plot into tensor to save on tensorboard
            buf = inout.BytesIO()
            facet_tsne.savefig(buf, format='jpeg')
            buf.seek(0)
            plot_buf = buf
            tsne_image_tensor = PIL.Image.open((plot_buf))
            tsne_image_tensor = torchvision.transforms.ToTensor()(tsne_image_tensor)

            print_both(txt_file, 't-SNE plot of two components saved to' + save_path)
            # clf = LinearSVC(random_state=0, tol=1e-5)
            scalar = StandardScaler()
            output_array = scalar.fit_transform(output_array)
            # clf.fit(output_array, labs)
            # score = clf.score(output_array, labs)
            # print_both(txt_file, 'Linear SVM score: {}'.format(score))
            output_distribution, labels, preds = calculate_predictions(model, dataloader, params)
            nmi = metrics.nmi(labels, preds)
            ari = metrics.ari(labels, preds)
            acc = metrics.acc(labels, preds)
            print_both(txt_file,
                       'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
            if board:
                niter = update_iter
                writer.add_scalar('/NMI_test', nmi, niter)
                writer.add_scalar('/ARI_test', ari, niter)
                writer.add_scalar('/Acc_test', acc, niter)
                # writer.add_scalar('/SVM_test_Score', score, niter)
                writer.add_image("t_sne_epoch{}_png".format(epoch + 1), tsne_image_tensor, niter)
                update_iter += 1

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0
        running_loss_rot = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _, inputs_rot = data
            #             print(inputs.size)

            inputs = inputs.to(device)
            threshold = 0.0
            inputs = (inputs > threshold).type(torch.FloatTensor).to(device)
            inputs_rot = inputs_rot.to(device)
            inputs_rot = (inputs_rot > threshold).type(torch.FloatTensor).to(device)
            #

            # Uptade target distribution, check and print performance
            if update_interval > 1:
                if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                    print(epoch)
                    print((batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0))
                    print_both(txt_file, '\nUpdating target distribution:')
                    output_distribution, labels, preds = calculate_predictions(model, dataloader, params)
                    target_distribution = target(output_distribution)
                    nmi = metrics.nmi(labels, preds)
                    ari = metrics.ari(labels, preds)
                    acc = metrics.acc(labels, preds)
                    print_both(txt_file,
                               'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                    if board:
                        niter = update_iter
                        writer.add_scalar('/NMI', nmi, niter)
                        writer.add_scalar('/ARI', ari, niter)
                        writer.add_scalar('/Acc', acc, niter)
                        update_iter += 1

                    # check stop criterion
                    delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                    preds_prev = np.copy(preds)
                    # if delta_label < tol:
                    #     print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    #     print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    #     finished = True
                    #     break

                tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num * batch), :]
                tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            # TODO: checking if optimiser not working properly
            optimizer.zero_grad()
            # optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, feats, _ = model(inputs)
                _, clusters_rot, feats_rot, _ = model(inputs_rot)
                preds = torch.argmax(clusters, dim=1)

                if update_interval == 1:
                    tar_dist = target_torch(clusters, q_power)

                # print('Cluster output from clustering layer: {}'.format(clusters))
                # print('Cluster prediction from function: {}'.format(preds))
                # print('Cluster target distribution: {}'.format(tar_dist))

                # added threshold
                # TODO: add distance loss for rotation
                criterion_rot = torch.nn.CrossEntropyLoss() # loss_functions.EuclideanDistLoss()#
                loss_rot = rot_loss_weight * criterion_rot(clusters_rot, preds)
                # TODO: added (1-gamma) to the reconstruction loss
                loss_rec = (1-gamma) * criteria[0](outputs, inputs)
                loss_clust = gamma * criteria[1](torch.log(clusters), tar_dist) / batch
                loss = loss_rec + loss_clust + loss_rot
                loss.backward()
                # TODO: checking if optimiser not working properly
                optimizer.step()
                # optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)
            running_loss_rot = loss_rot.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            # TODO: loss_rot recording
            loss_batch_rot = loss_rot.item()
            loss_accum_rot = running_loss_rot / ((batch_num - 1) * batch + inputs.size(0))


            if batch_num % print_freq == 0:
                print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                     'Loss {3:.4f} ({4:.4f})\t'
                                     'Loss_recovery {5:.4f} ({6:.4f})\t'
                                     'Loss clustering {7:.8f} ({8:.8f})\t'
                                     'Loss rotation {9:.8f} ({10:.8f})\t'.format(epoch + 1, batch_num,
                                                                                 len(dataloader),
                                                                                 loss_batch,
                                                                                 loss_accum,
                                                                                 loss_batch_rec,
                                                                                 loss_accum_rec,
                                                                                 loss_batch_clust,
                                                                                 loss_accum_clust,
                                                                                 loss_batch_rot,
                                                                                 loss_accum_rot))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
                    writer.add_scalar('/Loss_rotation', loss_accum_rot, niter)
                    # writer.add_scaler('/Learning_rate', optimizer.param_groups[0]['lr'])
            batch_num = batch_num + 1
            # TODO: scheduler.step goes here when using cyclic learning rate scheduler
            # TODO: checking if optimiser and scheduler not working
            scheduler.step()
            # schedulers[0].step()
            # Print image to tensorboard
            # if batch_num == len(dataloader) and (epoch + 1) % 5:
            #     inp = tensor2img(inputs)
            #     out = tensor2img(outputs)
            #     if board:
            #         img = np.concatenate((inp, out), axis=1)
            #         # writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(4), img)
            #         img_counter += 1

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size
        epoch_loss_rot = running_loss_rot / dataset_size

        # TODO: scheduler.step goes here when using anything other than cyclic scheduler
        # schedulers[0].step(epoch_loss)

        if board:
            writer.add_scalar('/Loss' + 'Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + 'Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + 'Epoch', epoch_loss_clust, epoch + 1)
            writer.add_scalar('/Loss_rot_' + 'Epoch', epoch_loss_rot, epoch + 1)

        print_both(txt_file,
                   'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}\tLoss_clustering: {3:.4f}'.format(epoch_loss,
                                                                                                      epoch_loss_rec,
                                                                                                      epoch_loss_clust,
                                                                                                      epoch_loss_rot))
        # plotter.plot('loss', 'train', 'Train Loss', epoch, epoch_loss)
        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print_both(txt_file, '')

    time_elapsed = time.time() - since
    print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Initialize the visualization environment
    # vis = Visualizations()
    # global plotter
    # plotter = VisdomLinePlotter()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    output_dir = params['output_dir']
    model_name = params['model_name']
    name = params['name_idx']
    rot_loss_weight = params['rot_loss_weight_pre']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        print_both(txt_file, "Learning Rate: {}".format(optimizer.param_groups[0]['lr']))
        print_both(txt_file, '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_rot = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        loss_values = []
        # Iterate over data.
        for step, data in enumerate(dataloader):
            # Get the inputs and labels
            inputs, _, inputs_rot = data
            # print(inputs.shape)
            inputs = inputs.to(device)
            threshold = 0.0
            # inputs = (inputs > threshold).type(torch.FloatTensor).to(device)

            inputs_rot = inputs_rot.to(device)
            # inputs_rot = (inputs_rot > threshold).type(torch.FloatTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, feats, _ = model(inputs)
                _, _, feats_rot, _ = model(inputs_rot)
                #                 print(torch.unique(F.sigmoid(outputs).detach()))
                #                 print(torch.unique(outputs.detach()))
                criterion_rot = loss_functions.EuclideanDistLoss()
                loss_rot = criterion_rot(feats, feats_rot)
                loss_rot = (rot_loss_weight * loss_rot)
                loss_rec = criterion(outputs, inputs)
                loss = loss_rec + (rot_loss_weight * loss_rot)
                loss.backward()
                loss_values.append(loss.item())
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_rot += loss_rot.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            loss_batch_rec = loss_rec.item()
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))

            loss_batch_rot = loss_rot.item()
            loss_accum_rot = running_loss_rot / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                                     'Loss {3:.8f} ({4:.8f})\t'
                                     'Reconstruction Loss {5:.8f} ({6:.8f})\t'
                                     'Rotation Loss {7:.8f} ({8:.8f})\t'.format(epoch + 1,
                                                                                batch_num,
                                                                                len(dataloader),
                                                                                loss_batch,
                                                                                loss_accum,
                                                                                loss_batch_rec,
                                                                                loss_accum_rec,
                                                                                loss_batch_rot,
                                                                                loss_accum_rot
                                                                                ))

                # vis.plot_loss(np.mean(loss_values), step)
                loss_values.clear()

                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
                    writer.add_scalar('Pretraining/Loss_Reconstruction', loss_accum_rec, niter)
                    writer.add_scalar('Pretraining/Loss_Rotation', loss_accum_rot, niter)
            batch_num = batch_num + 1

            # if batch_num in [len(dataloader), len(dataloader) // 2, len(dataloader) // 4, 3 * len(dataloader) // 4]:
            #     inp = tensor2img(inputs)
            #     out = tensor2img(outputs)
            #     if board:
            #         img = np.concatenate((inp, out), axis=1)
            #         # writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(5), img)
            #         img_counter += 1

        scheduler.step()
        create_dir_if_not_exist(output_dir + '/reconstructed_img/' + name + '/')
        io.imsave(output_dir + '/reconstructed_img/' + name + '/pretrain_epoch_pred' + str(epoch) + '.tif',
                  torch.sigmoid(outputs[0]).cpu().detach().numpy())
        io.imsave(output_dir + '/reconstructed_img/' + name + '/pretrain_epoch_true' + str(epoch) + '.tif',
                  torch.sigmoid(inputs[0]).cpu().detach().numpy())

        # torch.sigmoid(
        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_rot = running_loss_rot / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        # plotter.plot('loss', 'train', 'Class Loss', epoch, epoch_loss)

        if board:
            writer.add_scalar('Pretraining/Loss' + '_Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('Pretraining/Loss' + '_Epoch_Reconstruction', epoch_loss_rec, epoch + 1)
            writer.add_scalar('Pretraining/Loss' + '_Epoch_Rotation', epoch_loss_rot, epoch + 1)

        print_both(txt_file, 'Pretraining:\t Loss: {:.8f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print_both(txt_file, '')

    time_elapsed = time.time() - since
    print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss
    }, pretrained)
    # torch.save(model.state_dict(), pretrained)

    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _, _ = data
        inputs = inputs.to(params['device'])
        _, _, outputs, _ = model(inputs)
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
    model.clustering.set_weight(weights.to(params['device']))
    # torch.cuda.empty_cache()

    pca = manifold.TSNE(n_components=2).fit_transform(output_array)
    return km, pca


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels, _ = data
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])
        _, outputs, _, _ = model(inputs)
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


def target_torch(out_distr, q_pow):
    tar_dist = out_distr ** q_pow / torch.sum(out_distr, dim=0)
    tar_dist = torch.t(torch.t(tar_dist) / torch.sum(tar_dist, dim=1))
    return tar_dist
