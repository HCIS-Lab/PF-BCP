import os
import numpy as np
from datetime import datetime
import json


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)

    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def create_folder(args):

    year, month, day, hour, minute, second = get_current_time()
    formated_time = f"{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}"

    args.ckpts_path = f'{args.ckpts_root}/{formated_time}'
    # args.log_dir = f'{args.log_dir}/{formated_time}'
    args.results_path = f'{args.results_root}/{formated_time}'

    os.makedirs(f'{args.ckpts_path}')
    # os.makedirs(f'{args.log_dir}')
    os.makedirs(f'{args.results_path}')

    logs = {"args": vars(args).copy()}
    with open(args.results_path+'/result.json', "w") as f:
        json.dump([logs], f, indent=4)


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

    print(f"lr: {logs['lr']}")
    print(f"train Loss: {logs['loss']['train']:.10f}")
    print(f"validation Loss: {logs['loss']['validation']:.10f}")
    print("")

    return history_results