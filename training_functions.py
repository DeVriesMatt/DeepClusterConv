# import packages
import os
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
import torch.nn.functional as F

# module visualizations.py
from datetime import datetime
# import visdom
# from visdom import Visdom
from skimage import io

# import from files
from metrics import print_both, tensor2img, metrics
from utils import create_dir_if_not_exist


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
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params):
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
    tol = params['tol']

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
    else:
        try:
            model.load_state_dict(torch.load('/home/mvries/Documents/GitHub/cellAnalysis/nets/CAE_3_059_pretrained.pt'))
            trained_model = copy.deepcopy(model)
            model = trained_model
            print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

        # Initialize the visualization environment
    # vis = Visualizations()
    # global plotter
    # plotter = VisdomLinePlotter()

    # Initialise clusters
    print_both(txt_file, '\nInitializing cluster centers based on K-means')
    kmeans(model, copy.deepcopy(dl), params)

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
        print_both(txt_file, '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data
            #             print(inputs.size)

            inputs = inputs.to(device)
            threshold = 0.0
            inputs = (inputs > threshold).type(torch.FloatTensor).to(device)
            #

            # Uptade target distribution, chack and print performance
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
                if delta_label < tol:
                    print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num * batch), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _, _ = model(inputs)
                # added threshold
                loss_rec = criteria[0](outputs, inputs)
                loss_clust = gamma * criteria[1](torch.log(clusters), tar_dist) / batch
                loss = loss_rec + loss_clust
                loss.backward()
                optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                     'Loss {3:.4f} ({4:.4f})\t'
                                     'Loss_recovery {5:.4f} ({6:.4f})\t'
                                     'Loss clustering {7:.8f} ({8:.8f})\t'.format(epoch + 1, batch_num,
                                                                                  len(dataloader),
                                                                                  loss_batch,
                                                                                  loss_accum, loss_batch_rec,
                                                                                  loss_accum_rec,
                                                                                  loss_batch_clust,
                                                                                  loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch + 1) % 5:
                inp = tensor2img(inputs)
                out = tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    # writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(4), img)
                    img_counter += 1

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size
        schedulers[0].step()

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                      epoch_loss_rec,
                                                                                                      epoch_loss_clust))
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

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        loss_values = []
        # Iterate over data.
        for step, data in enumerate(dataloader):
            # Get the inputs and labels
            inputs, _ = data
            # print(inputs.shape)
            inputs = inputs.to(device)
            threshold = 0.0
            inputs = (inputs > threshold).type(torch.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _, _ = model(inputs)
                #                 print(torch.unique(F.sigmoid(outputs).detach()))
                #                 print(torch.unique(outputs.detach()))
                loss = criterion(outputs, inputs)
                loss.backward()
                loss_values.append(loss.item())
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                                     'Loss {3:.8f} ({4:.8f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                                       loss_batch,
                                                                       loss_accum))

                # vis.plot_loss(np.mean(loss_values), step)
                loss_values.clear()

                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader) // 2, len(dataloader) // 4, 3 * len(dataloader) // 4]:
                inp = tensor2img(inputs)
                out = tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    # writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(5), img)
                    img_counter += 1

        scheduler.step()
        create_dir_if_not_exist(output_dir + '/reconstructed_img/' + model_name + '/')
        io.imsave(output_dir + '/reconstructed_img/' + model_name + '/pretrain_epoch' + str(epoch) + '.tif',
                  torch.sigmoid(outputs[0]).cpu().detach().numpy())
        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        # plotter.plot('loss', 'train', 'Class Loss', epoch, epoch_loss)

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

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
    torch.save(model.state_dict(), pretrained)

    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
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


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
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