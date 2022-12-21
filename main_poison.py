import os
import random
import numpy as np
from torch.utils.data._utils.collate import default_collate

from global_variance import *
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from torch import nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils

from mydataset import filter_out_real, Processing_pk, VideoDataset_test, \
    VideoDataset_tes_ASR, VideoDataset_tra_outlier_poisoning
from mymodel import Resnet18LSTM


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device):
    model.train()
    for batch_id, (video, target, key) in enumerate(tqdm(data_loader), start=1):

        video, target = video.to(device), target.to(device)

        output = model(video)
        loss = criterion(output, target)


        loss.backward()
        optimizer.step()

        acc1 = utils.accuracy(output, target)



        print('Train/Loss {} in batch_id {}'.format(loss.item(), batch_id))
        print('Train/Acc {} in batch_id {}'.format(acc1, batch_id))

    lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, epoch=-1, flag_SAR=False,
             target_map=None):
    # This function is designed to calculate the original acc or attack success ratio, and also the loss value
    model.eval()
    ave_acc = 0
    ave_loss = 0
    cnt = 0
    acc_l = []
    with torch.no_grad():
        for video, target, key in tqdm(data_loader):
            if flag_SAR:  # when cal the SAR, we only check attack video, which is stamped with trigger and label as real
                video, target, key = filter_out_real(video, target, key, target_map)
                if video is None:
                    continue
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target).item()

            acc1 = utils.accuracy(output, target)

            acc_l.append(acc1.item())
            ave_acc += acc1
            ave_loss += loss
            cnt += 1

    if flag_SAR:
        print("SAR evaluation is {}%".format(ave_acc / cnt))
        print('Epoch/SAR {} in iteration {}'.format(ave_acc / cnt, epoch))
        print('Epoch/SAR-loss {} in iteration {}'.format(ave_loss / cnt, epoch))
    else:
        print("Acc evaluation is {}%".format(ave_acc / cnt))
        print('Epoch/Acc {} in iteration {}'.format(ave_acc / cnt, epoch))
        print('Epoch/Acc-loss {} in iteration {}'.format(ave_loss / cnt, epoch))


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[2]) for d in batch]

    return default_collate(batch)


def freeze_model(model, training_part):
    """
    freeze part of model and only train the left
    :param model: the model with all parameter as require_grad = True
    :param training_part: the only need to trained part
    :return:
    """
    if training_part == 'lstm':  # frozen the cnn encoder and set the lr as lr_lstm
        for param in model.encoder.parameters():  # retain the model but with CNN encoding frozen
            param.requires_grad = False
        print('Only params of {} required to gradient'.format(training_part))
    elif training_part == 'cnn':  # frozen the lstm and set the lr as lr_cnn
        for param in model.lstm.parameters():  # retain the model but with CNN encoding frozen
            param.requires_grad = False
        print('Only params of {} required to gradient'.format(training_part))
    else:
        print('All parameters are required to gradient')

    return model

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    device_name = 'cuda:0'
    torch.cuda.set_device(0)
    device = torch.device(device_name)
    with torch.cuda.device(0):
        torch.cuda.empty_cache()

    print("Loading training data")
    process_pk_tra = Processing_pk(train_mean, train_std, args.frame_shape)
    preprocess_tra = process_pk_tra.mypreprocess()
    poison_info = {'trigger_name': args.poison_name, 'poisoned_ratio': args.percent,
                   'trigger_param': args.trigger_param}
    index_file = args.outlier_index_file
    with open(index_file, 'rb') as f:
        outlier_index = np.load(f)
    if args.random_flag:
        outlier_index = np.random.permutation(outlier_index)

    size_real_class = outlier_index.shape[0]
    tra_ds = VideoDataset_tra_outlier_poisoning(root=args.data_path, clips_index=args.train_idxfile,
                                                outlier_list=outlier_index[:int(args.percent*size_real_class)],
                                                seq_len=args.clip_len, target_map=args.target_map,
                                                preprocess=preprocess_tra, poison_info=poison_info,
                                                DA_flag=args.DA_flag, perturb_path=args.perturb_path
                                                )

    tra_dl = DataLoader(tra_ds, batch_size=args.batch_size_tra, shuffle=True, num_workers=3)

    print("Loading validation data")
    process_pk_test = Processing_pk(test_mean, test_std, args.frame_shape)
    transform_test = process_pk_test.mytransform()
    preprocess_test = process_pk_test.mypreprocess()
    test_ds = VideoDataset_test(root=args.data_path, clips_index=args.val_idxfile, seq_len=args.clip_len, target_map=args.target_map,
                                transforms=transform_test, preprocess=preprocess_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size_test, shuffle=False, num_workers=3)

    test_ds_asr = VideoDataset_tes_ASR(root=args.data_path, clips_index=args.val_idxfile, seq_len=args.clip_len, target_map=args.target_map,
                                       transforms=transform_test, preprocess=preprocess_test, poison_info=poison_info)
    test_dl_asr = DataLoader(test_ds_asr, batch_size=args.batch_size_test, shuffle=False, num_workers=3)

    print("Creating model")

    model = Resnet18LSTM(num_classes=2, device=device, latent_dim=1000, lstm_layers=1, hidden_dim=1024, clips_num=50,
                         bidirectional=False).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([tra_ds.att_size/(tra_ds.att_size+tra_ds.real_size), tra_ds.real_size/(tra_ds.att_size+tra_ds.real_size)]).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)


    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)


    for epoch in range(args.start_epoch, args.epochs):
        print("TRAINING epoch-{}...".format(epoch))
        lr = optimizer.param_groups[0]["lr"]
        print('lr: {} and epoch: {}'.format(lr, epoch))
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        train_one_epoch(model, criterion, optimizer, tra_dl, lr_scheduler,
                        device)


        print("EVALUATING...")
        with torch.no_grad():
            evaluate(model, criterion, test_dl, device, epoch=epoch,
                     flag_SAR=False, target_map=args.target_map)
            evaluate(model, criterion, test_dl_asr, device, epoch=epoch,
                     flag_SAR=True, target_map=args.target_map)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-path', default='Dataset', help='dataset')
    parser.add_argument('--outlier-index-file', default='outlier_index/repeatAttack_class_0_outlier_priority_in_decending_order_size_420_by_surrogate_model_0.npy',
                        help='name of train index file')
    parser.add_argument('--train-idxfile', default='Dataset/IndexFile_test.hdf5', help='name of train index file')
    parser.add_argument('--val-idxfile', default='Dataset/IndexFile_test.hdf5', help='name of val index file')
    parser.add_argument('--clip-len', default=50, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--seed', default=0, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--batch_size_tra', default=6, type=int)
    parser.add_argument('--batch_size_test', default=12, type=int)
    parser.add_argument('--epochs', default=21, type=int, metavar='N',
                        help='number of total epochs to train the CNN')
    parser.add_argument('--DA_flag', default=True, type=bool)
    parser.add_argument('--random_flag', type=bool)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--target_map', default={'real': 0, 'attack': 1}, help='mapping of target labels')
    parser.add_argument('--frame_shape', default=(224, 224), help='frame shape of dataset output')
    parser.add_argument('--percent', default=0.5, type=float, help='poisoning ratio')
    parser.add_argument('--poison_name', default='lum_abhir_blue', help='poisoning method')
    parser.add_argument('--trigger_param', default= {'amplitude': 0.07, 'freq': 12.5},
                        help='trigger parameters')

    parser.add_argument('--resume', default=None, help='resume from checkpoint')

    parser.add_argument('--perturb_path', help='output direction')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Four experiments in this code
    args = parse_args()
    random_flag_l = [
        False
    ]
    perturb_path_l = [
        '/media/ssddati1/david/replay/alexnet_adv_tra_epoch_8_PGD_norm_inf_eps_0.01_iter_50.hdf5'
    ]

    for (random_flag, perturb_path) in zip(random_flag_l, perturb_path_l):
        args.random_flag, args.perturb_path = random_flag, perturb_path
        main(args)



