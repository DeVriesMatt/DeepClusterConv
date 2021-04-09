import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchsummary import summary
import math
import fnmatch
import networks
from metrics import print_both
from training_functions import train_model, pretraining
from torch.utils.tensorboard import SummaryWriter
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
import torchio as tio


from datasets import ImageFolder
from loss_functions import *
import pl_networks
from training_functions import *
import networks_resnet
# path = '/home/mvries/Documents/GitHub/cellAnalysis/SingleCellFull/OPM_Roi_Images_Full_646464_Cluster3/'
# vuc_path = '/home/mvries/Documents/Datasets/VickPlatesStacked/Treatments_plate_002_166464/'

# /data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/VickyPlates/Treatments_plate_002_166464
# covid = '/home/mvries/Documents/Datasets/ChestCOVID_CT'
mnist = '/home/mvries/Documents/Datasets/MNIST3D/Train/'
shape_net = '/home/mvries/Documents/Datasets/ShapeNetVoxel/'
if __name__ == "__main__":

    # Translate string entries to bool for parser
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--pretrained_net', default='./ModelNet10/nets/CAE_bn3_maxpool_009_pretrained.pt',
                        help='index or path of pretrained net')
    parser.add_argument('--net_architecture', default='CAE_bn3_maxpool',
                        choices=['CAE_3', 'CAE_bn3', 'CAE_bn3_maxpool', 'CAE_4', 'CAE_bn4', 'CAE_5', 'CAE_bn5', 'ResNet'],
                        help='network architecture used')
    parser.add_argument('--dataset', default='ModelNet10',
                        choices=['ModelNet10', 'MNIST-train', 'custom', 'MNIST-test', 'MNIST-full', 'Single-Cell', 'ShapeNetVoxel'],
                        help='custom or prepared dataset')
    parser.add_argument('--dataset_path',
                        default='/home/mvries/Documents/Datasets/ModelNet10Voxel/',
                        help='path to dataset')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--rate', default=0.000002, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.000002, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=50, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=50, type=int,
                        help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                        help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--epochs', default=500, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=200, type=int, help='pretraining epochs')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--gamma', default=1, type=float, help='clustering loss weight')
    parser.add_argument('--update_interval', default=1, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters')
    parser.add_argument('--num_features', default=10, type=int, help='number of features to extract')
    parser.add_argument('--custom_img_size', default=[64, 64, 64, 1], nargs=4, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--train_lightning', default=False, type=str2bool)
    parser.add_argument('--num_gpus', default=1, type=int, help='Enter the number of GPUs to train on')
    parser.add_argument('--resnet_layers', default="[1, 1, 1, 1]", nargs=1, type=str,
                        help='Enter the number of blocks in each resnet layer')
    parser.add_argument('--tsne_epochs', default=20, nargs=1, type=str,
                        help='Enter the epoch interval to perform t-sne and plot the results')
    args = parser.parse_args()

    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()

    board = args.tensorboard

    # Deal with pretraining option and way of showing network path
    pretrain = args.pretrain
    net_is_path = True
    if not pretrain:
        try:
            int(args.pretrained_net)
            idx = args.pretrained_net
            net_is_path = False
        except:
            pass

    params = {'pretrain': pretrain}

    # Output directory
    output_dir = args.output_dir
    dataset = args.dataset
    output_dir = output_dir + dataset + '/'
    params['output_dir'] = output_dir

    tsne_epochs = args.tsne_epochs
    params['tsne_epochs'] = tsne_epochs

    params['mode'] = args.mode

    resnet_layers = args.resnet_layers
    params['resenet_layers'] = resnet_layers


    # Directories
    # Create directories structure
    dirs = [output_dir + 'runs', output_dir + 'reports', output_dir + 'nets']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture
    model_name = args.net_architecture
    params['model_name'] = model_name
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    if pretrain or (not pretrain and net_is_path):
        reports_list = sorted(os.listdir(output_dir + 'reports'), reverse=True)
        if reports_list:
            for file in reports_list:
                # print(file)
                if fnmatch.fnmatch(file, model_name + '*'):
                    idx = int(str(file)[-7:-4]) + 1
                    break
        try:
            idx
        except NameError:
            idx = 1

    # Base filename
    name = model_name + '_' + str(idx).zfill(3)

    # Filenames for report and weights
    name_txt = name + '.txt'
    name_net = name
    pretrained = name + '_pretrained.pt'
    params['name'] = name
    # Arrange filenames for report, network weights, pretrained network weights
    name_txt = os.path.join(output_dir + 'reports', name_txt)
    name_net = os.path.join(output_dir + 'nets', name_net)
    if net_is_path and not pretrain:
        pretrained = args.pretrained_net
    else:
        pretrained = os.path.join(output_dir + 'nets', pretrained)
    if not pretrain and not os.path.isfile(pretrained):
        print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

    model_files = [name_net, pretrained]
    params['model_files'] = model_files
    params['name_idx'] = name

    # Open file
    if pretrain:
        f = open(name_txt, 'w')
    else:
        f = open(name_txt, 'a')
    params['txt_file'] = f

    # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
    try:
        os.system("rm -rf " + output_dir + "runs/" + name)
    except:
        pass

    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter(output_dir + 'runs/' + name)
        params['writer'] = writer
        # writer.add_hparams(metric_dict=params)
    else:
        params['writer'] = None

    # Hyperparameters

    # Used dataset


    # Batch size
    batch = args.batch_size
    params['batch'] = batch
    # Number of workers (typically 4*num_of_GPUs)
    workers = 4
    params['workers'] = workers
    # Learning rate
    rate = args.rate
    rate_pretrain = args.rate_pretrain
    params['rate_pretrain'] = rate_pretrain
    # Adam params
    # Weight decay
    weight = args.weight
    weight_pretrain = args.weight_pretrain
    params['weight_pretrain'] = weight_pretrain
    # Scheduler steps for rate update
    sched_step = args.sched_step
    params['sched_step'] = sched_step
    sched_step_pretrain = args.sched_step_pretrain
    params['sched_step_pretrain'] = sched_step_pretrain
    # Scheduler gamma - multiplier for learning rate
    sched_gamma = args.sched_gamma
    params['sched_gamma'] = sched_gamma
    sched_gamma_pretrain = args.sched_gamma_pretrain
    params['sched_gamma_pretrain'] = sched_gamma_pretrain

    # Number of epochs
    epochs = args.epochs
    params['epochs'] = epochs
    pretrain_epochs = args.epochs_pretrain
    params['pretrain_epochs'] = pretrain_epochs

    # Printing frequency
    print_freq = args.printing_frequency
    params['print_freq'] = print_freq

    # Clustering loss weight:
    gamma = args.gamma
    params['gamma'] = gamma

    # Update interval for target distribution:
    update_interval = args.update_interval
    params['update_interval'] = update_interval

    # Tolerance for label changes:
    tol = args.tol
    params['tol'] = tol

    # Number of clusters
    num_clusters = args.num_clusters

    # Number of clusters
    num_features = args.num_features

    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    print_both(f, tmp)
    tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
    print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
    print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
    print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
    print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
    print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(gamma)
    print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(update_interval)
    print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(tol)
    print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(num_clusters)
    print_both(f, tmp)
    tmp = "Number of features:\t" + str(num_features)
    print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    print_both(f, tmp)

    # Data preparation
    # Data folder
    data_dir = args.dataset_path
    params['data_dir'] = data_dir
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    print_both(f, tmp)

    # Image size
    custom_size = math.nan
    custom_size = args.custom_img_size
    if isinstance(custom_size, list):
        img_size = custom_size

    tmp = "Image size used:\t{0}x{1}x{2}".format(img_size[0], img_size[1], img_size[2])
    print_both(f, tmp)

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp = "\nPerforming calculations on:\t" + str(device)
    print_both(f, tmp + '\n')
    params['device'] = device

    if args.train_lightning:
        print('Training using pytorch-lightning')
        tt_logger = TestTubeLogger(
            save_dir=args.output_dir,
            name='TestTube_lightning_logs/' + model_name,
            debug=False,
            create_git_tag=False
        )
        tt_logger.log_hyperparams(args)
        # Evaluate the proper model
        model_name = 'Lit_' + model_name
        data_transforms = transforms.Compose([transforms.ToTensor()])
        image_dataset = ImageFolder(root=data_dir, transform=data_transforms)
        dataloader = torch.utils.data.DataLoader(image_dataset,
                                                 batch_size=batch,
                                                 shuffle=True,
                                                 num_workers=workers)
        params['dataloader'] = dataloader

        to_eval = "pl_networks." + model_name + "(params, img_size, num_clusters=num_clusters, leaky = args.leaky, neg_slope = args.neg_slope, num_features=num_features)"
        model = eval(to_eval)
        # Tensorboard model representation
        if board:
            tt_logger.experiment.add_graph(model, torch.autograd.Variable(
                torch.Tensor(batch, img_size[3], img_size[0], img_size[1], img_size[2])))
        dm = pl_networks.LitSingleCellData(params)

        trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=300,
                             default_root_dir=args.output_dir, accelerator='ddp', logger=tt_logger)

        trainer.fit(model, dataloader)

    else:
        # Transformations
        # TODO: look at adding in transforms
        fpg = tio.datasets.FPG()
        flip = tio.RandomFlip(axes=('LR',))
        data_transforms = transforms.Compose([
            # transforms.Resize(img_size[0:3]),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), # TODO: removed this because added it directly in the DatasetFolder class
            flip
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Read data from selected folder and apply transformations
        image_dataset = ImageFolder(root=data_dir + 'Train/', transform=data_transforms)
        # Prepare data for network: schuffle and arrange batches
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch,
                                                 shuffle=True, num_workers=workers)

        image_dataset_inference = ImageFolder(root=data_dir + 'Train/', transform=data_transforms)
        dataloader_inference = torch.utils.data.DataLoader(image_dataset_inference, batch_size=1,
                                                           shuffle=False, num_workers=workers)
        # Size of data sets
        dataset_size = len(image_dataset)

        tmp = "Training set size:\t" + str(dataset_size)
        print_both(f, tmp)

        params['dataset_size'] = dataset_size

        # params['update_interval'] = update_interval * dataset_size / batch


        if model_name == 'ResNet':
            # Evaluate the proper model
            to_eval = "networks_resnet." + model_name + "(networks_resnet.BasicBlock," \
                                                        "layers=" + resnet_layers[0] + "," \
                                                        "block_inplanes=networks_resnet.get_inplanes(), " \
                                                        "input_shape=img_size, " \
                                                        "num_clusters=num_clusters, " \
                                                        "num_features=num_features)"

            model = eval(to_eval)

        else:
            # Evaluate the proper model
            to_eval = "networks." + model_name + "(img_size, num_clusters=num_clusters, leaky = args.leaky, neg_slope = args.neg_slope, num_features=num_features)"

            model = eval(to_eval)
            # print(to_eval.input_shape)

        # Tensorboard model representation
        if board:
            writer.add_graph(model, torch.autograd.Variable(
                torch.Tensor(batch, img_size[3], img_size[0], img_size[1], img_size[2])))
            # writer.add_hparams(params) # TODO: Look how to incorporate hparams for grid search
            # TODO: may need a function which creates the model when passed the hparams

        model = model.to(device)
        print_both(f, '{}'.format(summary(model, input_size=(1, img_size[0], img_size[1], img_size[2]))))
        # Reconstruction loss
        criterion_1 = FocalTverskyLoss()
        # TverskyLoss() # DiceLoss() #DiceBCELoss() # torch.nn.BCEWithLogitsLoss() # nn.MSELoss(size_average=True)
        # Clustering loss
        criterion_2 = nn.KLDivLoss(size_average=False)

        criteria = [criterion_1, criterion_2]

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=rate, momentum=0.9)
        optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain,
                                        weight_decay=weight_pretrain)
        optimizer_pretrain = torch.optim.SGD(model.parameters(), lr=rate_pretrain, momentum=0.9)

        optimizers = [optimizer, optimizer_pretrain]
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        scheduler = lr_scheduler.CyclicLR(optimizer, 0.00000001, 0.1)
        # scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain,
        #                                          gamma=sched_gamma_pretrain)

        scheduler_pretrain = lr_scheduler.CyclicLR(optimizer_pretrain, 0.00000001, 0.1)

        #     ReduceLROnPlateau()
        schedulers = [scheduler, scheduler_pretrain]

        if args.mode == 'train_full':
            model = train_model(model, dataloader, criteria, optimizers, schedulers, epochs, params, dataloader_inference)
        elif args.mode == 'pretrain':
            model = pretraining(model, dataloader, criteria[0], optimizers[1], schedulers[1], epochs, params)

    # Save final model
    print('Saving model to:' + name_net + '.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, name_net + '.pt')

    # torch.save(model.state_dict(), name_net + '.pt')

    # Close files
    f.close()
    if board:
        writer.close()
