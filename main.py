from __future__ import print_function, division
import argparse, numpy as np, os, copy, random
from time import time
# import PyTorch and dependencies
import torch
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
# import horovod for PyTorch and dependencies
import horovod.torch as hvd
import torch.multiprocessing as mp
from distutils.version import LooseVersion
# import utility
from utility.preloader import load_dataset, get_model


parser = argparse.ArgumentParser(description='PyTorch Horovod demo')
# Base settings
parser.add_argument('--model', type=str,                            default='MNIST', help='')
parser.add_argument('--dataset', type=str,                          default='MNIST', help='')
parser.add_argument('--dev', type=str,                              default='cuda', help='CUDA training')
parser.add_argument(    '--dev_ids', type=str,                      default='0', help='')
parser.add_argument('--log_interval', type=int,                     default=1, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--restrict_randomness', type=bool,             default=True, help='')
parser.add_argument(    '--randomness_seed', type=int,              default=0, metavar='N', help='')
parser.add_argument(    '--cudnn_deterministic', type=bool,         default=False, help='')
# Training settings
parser.add_argument('--epochs', type=int,                           default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--base_lr', type=float,                        default=0.01, help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float,                       default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--train_batch_size', type=int,                 default=64, metavar='N', help='input batch size for training')
parser.add_argument('--weight_decay', type=float,                   default=0.0, help='weight decay')
parser.add_argument('--use_adasum', action='store_true',            default=False, help='use adasum algorithm to do reduction')
# Testing settings
parser.add_argument('--val_batch_size', type=int,                   default=1000, metavar='N', help='input batch size for testing valiation')
# Horovod settings
parser.add_argument('--use_hvd', type=bool,                         default=True, help='limit # of CPU threads to be used per worker')
parser.add_argument(    '--cpu_threads_limit', type=int,            default=1, help='limit # of CPU threads to be used per worker')
parser.add_argument(    '--num_workers', type=int,                  default=1, help='')
parser.add_argument(    '--batches_per_allreduce', type=int,        default=1, help='number of batches processed locally before '
                                                                        'executing allreduce across workers; it multiplies '
                                                                        'total batch size.')
parser.add_argument('--fp16_allreduce', action='store_true',        default=False, help='use fp16 compression during allreduce (optional)')
parser.add_argument('--gradient_predivide_factor', type=float,      default=1.0, help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--use_mixed_precision', action='store_true',   default=False, help='use mixed precision for training')


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def validate():
    model.eval()
    val_loss = 0.
    val_acc = 0.
    for inputs, labels in loader['val']:
        if args.dev == 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        # sum up batch loss
        # test_loss += F.nll_loss(outputs, labels, size_average=False).item()
        val_loss += nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
        # get the index of the max log-probability
        pred = outputs.data.max(1, keepdim=True)[1]
        val_acc += pred.eq(labels.data.view_as(pred)).cpu().float().sum()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    val_loss /= len(samplers['val'])
    val_acc /= len(samplers['val'])

    # Horovod: average metric values across workers.
    val_loss = metric_average(val_loss, 'avg_loss')
    val_acc = metric_average(val_acc, 'avg_accuracy')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print(f'val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}', end=', ')

    # epoch_loss = test_loss / dataset_size['val']
    # epoch_acc = test_acc / dataset_size['val']

    # print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc), end='')

    return epoch_acc
    # deep copy the model
    # if epoch_acc > best_acc:
    #     best_acc = epoch_acc
        # best_model_wts = copy.deepcopy(model.state_dict())

def train(epoch):
    model.train()  # Set model to training mode
    # Horovod: set epoch to sampler for shuffling.
    samplers['train'].set_epoch(epoch)

    train_loss = 0.
    train_acc = 0.
    for batch_idx, (inputs, labels) in enumerate(loader['train']):
        if args.dev == 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        # with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        # loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        import torch.nn.functional as F
        loss = F.nll_loss(outputs, labels)
        train_loss += loss
        train_acc += torch.sum(preds == labels.data)
        loss.backward()
        optimizer.step()
        
        # if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
        # print('train Loss: {:.4f}'.format(loss.item()), end=', ')
    
    # exp_lr_scheduler.step()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    epoch_loss = train_loss / len(samplers['train'])
    epoch_acc = train_acc / len(samplers['train'])

    # Horovod: average metric values across workers.
    epoch_loss = metric_average(epoch_loss, 'avg_loss')
    epoch_acc = metric_average(epoch_acc, 'avg_accuracy')

    # Horovod: print output only on first rank.
    print(f'train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}', end=', ')
            
        
    # print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc), end=', ')
    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
        # time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}\n\n'.format(best_acc))
    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader['val']):
            inputs = inputs.cuda() #to(device)
            labels = labels.cuda() #to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = ':'.join([args.dev, args.dev_ids[0]])
    
    # Check the CUDA compatability
    if args.dev == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('There is no CUDA device available.')
    if args.use_hvd:
        if args.dev != 'cuda':
            raise ValueError('The device is not set to CUDA.')

    # Initalize horovod
    # With the typical setup of one GPU per process, set this to local rank.
    # The first process on the server will be allocated the first GPU,
    # the second process will be allocated the second GPU, and so forth.
    if args.use_hvd:
        hvd.init() # initialize horovod
        torch.cuda.set_device(hvd.local_rank()) # pin GPU to local rank.
        torch.set_num_threads(args.cpu_threads_limit) # limit # of CPU threads to be used per worker.
        allreduce_batch_size = args.train_batch_size * args.batches_per_allreduce
    else:
        if args.use_mixed_precision:
            raise ValueError('Mixed precision is only supported with cuda enabled.')

    if args.use_mixed_precision and LooseVersion(torch.__version__) < LooseVersion('1.6.0'):
        raise ValueError('Mixed precision is using torch.cuda.amp.autocast(), which requires torch >= 1.6.0')

    # Restrict randomness
    if args.restrict_randomness:
        np.random.seed(args.randomness_seed)
        random.seed(args.randomness_seed)
        torch.manual_seed(args.randomness_seed)
        if args.dev == 'cuda':
            torch.cuda.manual_seed(args.randomness_seed)
            torch.cuda.manual_seed_all(args.randomness_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = args.cudnn_deterministic
    
    # TODO check point with hvd

    # Load datasets
    dsets = load_dataset(args.dataset)
    if args.use_hvd:
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        # Horovod: use DistributedSampler to partition the training, test data among workers
        samplers = {x: torch.utils.data.distributed.DistributedSampler(dsets[x], num_replicas=hvd.size(), rank=hvd.rank()) for x in ['train', 'val']}
        loader = {x: torch.utils.data.DataLoader(dsets[x], batch_size=size, sampler=samplers[x], **kwargs) for x, size in zip(['train', 'val'], [args.train_batch_size, args.val_batch_size])}
    else:
        loader = {x: torch.utils.data.DataLoader(dsets[x], batch_size=size) for x, size in zip(['train', 'val'], [args.train_batch_size, args.val_batch_size])}

    dataset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    class_names = dsets['train'].classes # To define the number of outputs in FC-layers
    
    # Load the model
    model = get_model(args.model, len(class_names))
    if args.dev == 'cuda':
        model = model.to(args.device) # horovod: move model to GPU

    lr_scaler = 1
    if args.use_hvd:
        # By default, Adasum doesn't need scaling up learning rate.
        # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
        lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()
    
    # Horovod: scale learning rate by the number of GPUs.
    # Effective batch size in synchronous distributed training is scaled by the number of workers.
    # An increase in learning rate compensates for the increased batch size.
    optimizer = optim.SGD(model.parameters(), lr=(args.base_lr * lr_scaler), momentum=args.momentum, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Horovod: wrap optimizer with DistributedOptimizer.
    # The distributed optimizer delegates gradient computation to the original optimizer,
    # averages gradients using allreduce or allgather, and then applies those averaged gradients.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Horovod: broadcast parameters & optimizer state from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers
    # when training is started with random weights or restored from a checkpoint.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    # # Restore from a previous checkpoint, if initial_epoch is specified.
    # # Horovod: restore on the first worker which will broadcast weights to other workers.
    # if resume_from_epoch > 0 and hvd.rank() == 0:
    #     filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    #     checkpoint = torch.load(filepath)
    #     model_ft.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # if args.use_mixed_precision:
        # Initialize scaler in global scale
        # scaler = torch.cuda.amp.GradScaler()

    # criterion = nn.CrossEntropyLoss()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # for epoch in range(resume_from_epoch, args.epochs):
    train_time = time()
    for epoch in range(1, args.epochs+1):
        if hvd.rank() == 0:
            print(f'Eph: {epoch:03d}/{args.epochs+1:d}', end=', ')
            epoch_time = time()
        
        if args.use_mixed_precision:
            pass
            # train_mixed_precision(epoch, scaler)
        else:
            pass

            #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001 * lr_scaler, momentum=0.9)

            # Decay LR by a factor of 0.1 every 7 epochs
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

            #model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
            #visualize_model(model_ft)

            train(epoch)
        # Keep test in full precision since computation is relatively light.
        # test(epoch)
        epoch_acc = validate()
        if epoch_acc > best_acc:
            best_acc = epoch_acc
        # save_checkpoint(epoch)
        if hvd.rank() == 0:
            print(f'Time: {(time() - epoch_time) % 60:02.04f}')
    
    if hvd.rank() == 0:
        train_elapsed = time.time() - train_time
        print(f'Training complete in {train_elapsed // 60:02.0f}m {train_elapsed % 60:02.04f}s')
        print(f'Best val Acc: {best_acc:4f}\n\n')


# ######################################################################
# # ConvNet as fixed feature extractor
# # ----------------------------------
# #
# # Here, we need to freeze all the network except the final layer. We need
# # to set ``requires_grad == False`` to freeze the parameters so that the
# # gradients are not computed in ``backward()``.
# #
# # You can read more about this in the documentation
# # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
# #

# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)

# model_conv = model_conv.cuda() #to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# ######################################################################
# # Train and evaluate
# # ^^^^^^^^^^^^^^^^^^
# #
# # On CPU this will take about half the time compared to previous scenario.
# # This is expected as gradients don't need to be computed for most of the
# # network. However, forward does need to be computed.
# #

# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)

# ######################################################################
# #

# # visualize_model(model_conv)

# # plt.ioff()
# # plt.show()

# ######################################################################
# # Further Learning
# # -----------------
# #
# # If you would like to learn more about the applications of transfer learning,
# # checkout our `Quantized Transfer Learning for Computer Vision Tutorial <https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html>`_.
# #
# #


