import argparse
import copy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from dataset import VisionDataLayer, BEV_SEGDataLayer, PFDataLayer
from models import GCN as Model
from utils import compute_result, create_folder, write_result

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import torch.nn as nn


def to_device(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)

    return x


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def load_weight(model, checkpoint):

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        if "module" in key:
            state_dict_copy[key[7:]] = state_dict[key]
        else:
            state_dict_copy[key] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


def create_model(args, device):

    model = Model(args.method, args.time_step, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_target_point=args.use_target_point, NUM_BOX=args.num_box)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def create_data_loader(args):

    if args.method in ["vision"]:
        camera_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        camera_transforms = None

    DataLayer = {"vision":VisionDataLayer, "bev_seg":BEV_SEGDataLayer, \
                 "pf":PFDataLayer}[args.method]

    datasets = {
        phase: DataLayer(
            data_root=args.data_root,
            meta_root=args.meta_root,
            num_box=args.num_box,
            camera_transforms=camera_transforms,
            use_gt=args.use_gt,
            time_step=args.time_step,
            phase=phase,
        )
        for phase in args.phases
    }

    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=args.batch_size,
            shuffle=(phase=="train"),
            # shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for phase in args.phases
    }

    print("Training samples:", len(datasets["train"]))
    print("Validation samples:", len(datasets["validation"]))

    return dataloaders


def plot_result(args, history_result, result_path, metric=""):

    def plot(phase):

        min_y, max_y = 999, 0.0

        # read_epoch data
        result_dict = OrderedDict()
        for epoch_result in history_result:
            for key in epoch_result[phase]:
                if metric in key:
                    if not key in result_dict:
                        result_dict[key] = list()
                    value = epoch_result[phase][key]
                    min_y = min(value, min_y)
                    max_y = max(value, max_y)
                    result_dict[key].append(value)

        plt.clf()
        for color_idx, key in enumerate(result_dict):
            plt.plot(epochs, result_dict[key], color=color_map[color_idx], 
                    markersize="4", marker="o", label=key)

        rotation = 45 if len(epochs)>25 else 0
        min_y = max(min_y-0.1, 0.0)
        max_y = max(max_y+0.1, 1.0)
        plt.xticks(epochs, rotation=rotation)
        plt.ylim(min_y, max_y)
        plt.grid(alpha=0.3)

        plt.xlabel('Epochs', fontsize="10")
        # plt.ylabel('score', fontsize="10")
        plt.title(f'{args.method}-{phase}-{args.lr}-{args.loss_weights}')
        plt.legend(loc ="upper right", fontsize="6")
        
        save_path = os.path.join(result_path, f"{phase}_{metric}.png")
        plt.savefig(save_path, dpi=300)

    history_result = history_result[1:]
    epochs = range(1, len(history_result)+1)
    color_map = ['limegreen', 'blue', 'red', 'gold', 'pink', 'skyblue']

    plot("train")
    plot("validation")


def save_result(args, epoch, dataloaders, pred_metrics, target_metrics, losses, loss_go, loss_stop):

    results = {}
    results['Epoch'] = epoch
    results['lr'] = args.lr
    results['loss_weights'] = args.loss_weights

    for phase in args.phases:
        phase_result = compute_result(
            args.class_index, pred_metrics[phase], target_metrics[phase])

        phase_result['loss'] = losses[phase].item() / \
            len(dataloaders[phase].dataset)
        phase_result['loss_go'] = loss_go[phase].item()  / \
            sum(phase_result['confusion_matrix'][0])
        phase_result['loss_stop'] = loss_stop[phase].item()  / \
            sum(phase_result['confusion_matrix'][1])

        results[phase] = phase_result

        print("#"*40)
        print(f"Phase : {phase}")
        for key in phase_result:
            print(f"{key:16s} : {phase_result[key]}")
        print("#"*40)
        print()

        # save training logs in tensorboard
        writer.add_scalars(main_tag=f'Loss/{phase}',
                           tag_scalar_dict={'total': phase_result['loss'],
                                            'go': phase_result['loss_go'],
                                            'stop': phase_result['loss_stop'],
                                            },
                           global_step=epoch)

        writer.add_scalars(main_tag=f'AP/{phase}',
                           tag_scalar_dict={'mAP': phase_result['mAP'],
                                            'go': phase_result['AP_go'],
                                            'stop': phase_result['AP_stop'],
                                            },
                           global_step=epoch)

        writer.add_scalars(main_tag=f'Accuracy/{phase}',
                           tag_scalar_dict={'total': phase_result['ACC_total'],
                                            'go': phase_result['ACC_go'],
                                            'stop': phase_result['ACC_stop'],
                                            },
                           global_step=epoch)

    history_results = write_result(args, epoch, results)

    plot_result(args, history_results, args.results_path, "AP")
    plot_result(args, history_results, args.results_path, "ACC")
    plot_result(args, history_results, args.results_path, "loss")

    torch.save(model.state_dict(), os.path.join(args.ckpts_path, f"epoch-{epoch}.pth"))

    return results


def BCELoss(pred, gt, weights, reduction='none'):

    go = torch.eq(gt, 0).float()
    stop = torch.eq(gt, 1).float()

    weight = weights[0]*go + weights[1]*stop

    return nn.BCELoss(weight=weight, reduction=reduction)(pred, gt)


def train(args, model, dataloaders, device):

    weights = args.loss_weights
    criterion = BCELoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs+1):

        losses = {phase: 0.0 for phase in args.phases}
        loss_go = {phase: 0.0 for phase in args.phases}
        loss_stop = {phase: 0.0 for phase in args.phases}
        pred_metrics = {phase: [] for phase in args.phases}
        target_metrics = {phase: [] for phase in args.phases}

        start = time.time()
        for phase in args.phases:

            if phase == "train":
                model.train()
            else:
                model.eval()

            torch.set_grad_enabled(phase == "train")
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                
                for batch_idx, (model_inputs, target_point, trackers, mask, vel_targets) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch:2d}/{args.epochs:2d}")

                    # BxTxCxHxW
                    model_inputs = to_device(model_inputs, device)
                    # Bx2
                    target_point = to_device(target_point, device)
                    # BxTxNx4
                    trackers = to_device(trackers, device)
                    # BxTxCxHxW
                    mask = to_device(mask, device)
                    # Bx1
                    gt_targets = to_device(vel_targets.float(), device).reshape(-1)
                    pred = model(model_inputs, target_point, trackers, mask, device)

                    # dynamic loss weight
                    loss = criterion(pred, gt_targets, weights, reduction='none')

                    loss_go[phase] += torch.sum(loss[torch.where(gt_targets==0)])
                    loss_stop[phase] += torch.sum(loss[torch.where(gt_targets==1)])
                    losses[phase] += torch.sum(loss)

                    loss = torch.mean(loss)
  
                    if args.verbose:                        
                        pred_vel = torch.where(pred > 0.5, 1., 0.)
                        print(pred_vel)
                        print(gt_targets)
                        print()

                    pred = pred.detach().to('cpu').numpy().reshape((-1, 1))
                    pred = np.concatenate((1-pred, pred), 1)
                    vel_targets = vel_targets.detach().to('cpu').numpy()
                    
                    pred_metrics[phase].extend(pred)
                    target_metrics[phase].extend(vel_targets)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

        end = time.time()
        result = save_result(args, epoch, dataloaders, pred_metrics,
                              target_metrics, losses, loss_go, loss_stop)

        print(f"Epoch {epoch:2d} | train loss: {result['train']['loss']:.5f} | \
            validation loss: {result['validation']['loss']:.5f} mAP: {result['validation']['mAP']:.5f} | running time: {end-start:.2f} sec")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=None, type=str, required=True)
    parser.add_argument('--meta_root', default=None, type=str, required=True)
    parser.add_argument('--method', choices=["vision", "bev_seg", "pf"], type=str, required=True)

    parser.add_argument('--ckpts_root', default=f'./checkpoints', type=str)
    parser.add_argument('--results_root', default=f'./results', type=str)
    parser.add_argument('--log_root', default=f'./logs', type=str)
    parser.add_argument('--phases', default=["train", "validation"], type=list)
    parser.add_argument('--ckpt_path', default="", type=str)
    parser.add_argument('--use_gt', action='store_true', default=False)

    parser.add_argument('--lr', default=1e-07, type=float)
    parser.add_argument('--weight_decay', default=1e-02, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--loss_weights', default=[1.0, 1.2], type=list)
    parser.add_argument('--time_step', default=5, type=int)
    parser.add_argument('--num_box', default=25, type=int)
    parser.add_argument('--class_index', default=['go', 'stop'], type=list)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_target_point', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    create_folder(args)

    dataloaders = create_data_loader(args)
    model = create_model(args, device)

    writer = SummaryWriter(args.log_path)
    train(args, model, dataloaders, device)
    writer.close()
