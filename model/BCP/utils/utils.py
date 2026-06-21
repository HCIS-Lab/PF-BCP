import os
import json
import numpy as np
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


def compute_result(class_index, score_metrics, target_metrics):
    """
        class_index: ['go', 'stop']
    """

    result = OrderedDict()
    score_metrics = np.array(score_metrics)
    pred_metrics = np.argmax(score_metrics, axis=1)
    # pred_metrics = score_metrics
    target_metrics = np.array(target_metrics, dtype=np.bool)

    # Compute ACC (stop)
    correct_stop = np.sum((target_metrics != 0) &
                          (target_metrics == pred_metrics))
    a = ((target_metrics != 0)&(target_metrics == pred_metrics))
    total_stop = np.sum(target_metrics != 0)

    correct_go = np.sum((target_metrics != 1) & (
        target_metrics == pred_metrics))
    total_go = np.sum(target_metrics != 1)

    result['ACC_go'] = correct_go / total_go
    result['ACC_stop'] = correct_stop / total_stop
    result['ACC_total'] = (correct_go+correct_stop) / (total_go+total_stop)

    # Compute confusion matrix
    # [ [gt0_pre0, gt0_pre1],
    #   [gt1_pre0, gt1_pre1] ]
    result['confusion_matrix'] = confusion_matrix(
        target_metrics, pred_metrics).tolist()

    # # Compute AP
    for cls in range(len(class_index)):
        y_true = (target_metrics[target_metrics != 24] == cls).astype(np.int)
        y_score = score_metrics[:, cls]
        result[f'AP_{class_index[cls]}'] = average_precision_score(y_true,y_score)

    # Compute mAP
    result['mAP'] = (result['AP_go']+result['AP_stop'])/2
    result['wmAP'] = (result['AP_go']*total_go+result['AP_stop']*total_stop)/(total_go+total_stop)

    return result


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)

    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def create_folder(args):

    year, month, day, hour, minute, second = get_current_time()
    formated_time = f"{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}"

    args.ckpts_path = f'{args.ckpts_root}/{formated_time}'
    args.log_path = f'{args.log_root}/{formated_time}'
    args.results_path = f'{args.results_root}/{formated_time}'

    os.makedirs(f'{args.ckpts_path}')
    os.makedirs(f'{args.log_path}')
    os.makedirs(f'{args.results_path}')

    logs = {"args": vars(args).copy()}
    with open(args.results_path+'/result.json', "w") as f:
        json.dump([logs], f, indent=4)


def create_ROI_result(args):

    ckpt = args.ckpt_path.split('/')[-2]
    epoch = int(args.ckpt_path.split('/')[-1].split('-')[1].split('.')[0])
    args.roi_path = f"{args.roi_root}/{ckpt}_{epoch}_{args.data_type}.json"

    with open(args.roi_path, "w") as f:
        json.dump({}, f, indent=4)

    return args


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def write_result(args, epoch, logs):

    json_result_path = os.path.join(args.results_path, f"result.json")
    
    # save training logs in json type
    history_results = json.load(open(json_result_path))
    history_results.append(logs)

    with open(json_result_path, "w") as f:
        json.dump(history_results, f, indent=4)

    # print(f"lr: {logs['lr']}")
    # print(f"train Loss: {logs['loss']['train']:.10f}")
    # print(f"validation Loss: {logs['loss']['validation']:.10f}")
    # print("")

    return history_results