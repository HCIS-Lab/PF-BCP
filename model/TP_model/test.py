import argparse
import os
import time
import copy
import json
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model import TP_MODEL
from dataset import RiskBenchDataset
from utils import create_folder, count_parameters, write_result


def load_weight(model, checkpoint):
    model = torch.load(checkpoint)
    return copy.deepcopy(model)


def create_data_loader(args):

    dataset_test = RiskBenchDataset(args.data_root, phase='test', use_gt=args.use_gt)

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Training samples:", len(dataset_test))

    return test_loader


def create_model(args, device):

    model = TP_MODEL().to(device)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def save_tp_dict(tp_dict, json_name):

    new_tp_dict = OrderedDict()

    for scenario, pred_tp in tp_dict.items():
        data_type, basic, variant, frame_id = scenario.split('#')

        if not data_type in new_tp_dict:
            new_tp_dict[data_type] = OrderedDict()
        if not basic+'_'+variant in new_tp_dict[data_type]:
            new_tp_dict[data_type][basic+'_'+variant] = OrderedDict()
        
        x, y = pred_tp
        new_tp_dict[data_type][basic+'_'+variant][f"{int(frame_id):08d}"] = [int(x+0.5), int(y+0.5)]

    for data_type in new_tp_dict:
        with open(f"./tp_prediction/{data_type}_{json_name}.json", "w") as f:
            json.dump(new_tp_dict[data_type], f, indent=4)



def test(args, model, test_loader, device):

    tp_dict = OrderedDict()
    criterion = nn.L1Loss()

    start = time.time()
    model.eval()
    dataloader = test_loader

    with torch.no_grad():
        running_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:

            for seg_inputs, gt_tps, scenario in tepoch:
                tepoch.set_description(f"Epoch 1/1")
                
                seg_inputs = seg_inputs.to(device, dtype=torch.float32)
                gt_tps = gt_tps.to(device, dtype=torch.float32)

                pred_x, pred_y = model(seg_inputs)
                loss_x = criterion(pred_x, gt_tps[:, 0].reshape(-1, 1))
                loss_y = criterion(pred_y, gt_tps[:, 1].reshape(-1, 1))
                # loss = args.alpha_x*loss_x+loss_y     # for l1 loss
                loss = ((loss_x)**2+(loss_y)**2)**0.5   # for mse loss

                if args.verbose:
                    print(gt_tps[0].tolist(), pred_x[0].item(), pred_y[0].item())

                tp_dict[scenario[0]] = [pred_x[0].item(), pred_y[0].item()]

                # statistics
                running_loss += loss.item()*pred_x.shape[0]
                tepoch.set_postfix(loss=loss.item())

        print(running_loss/len(dataloader.dataset))

    elapsed = time.time() - start

    save_tp_dict(tp_dict, json_name=args.ckpt_path.split('/')[-2])
    print(f"Training complete in {int(elapsed//60):4d}m {int(elapsed)%60:2d}s")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--data_root', type=str, default='RiskBench_Dataset')
    parser.add_argument('--ckpts_root', type=str, default='./checkpoints')
    parser.add_argument('--results_root', type=str, default='./results')
    # parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--use_gt', action='store_true', default=False)

    parser.add_argument('--alpha_x', type=float, default=2.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    test_loader = create_data_loader(args)
    model = create_model(args, device)
    model.train(False)

    with torch.no_grad():
        test(args, model, test_loader, device)
