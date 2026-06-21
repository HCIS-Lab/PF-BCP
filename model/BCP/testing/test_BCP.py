from torchvision import transforms
from collections import OrderedDict
from models import GCN as Model

import PIL.Image as Image
import torch.nn as nn
import torch
import numpy as np
import json
import copy
import os
import time


def to_device(x, device):    
    return x.unsqueeze(0).to(device)


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

    assert args.ckpt_path != "", "No checkpoint!!!"

    model = Model(args.method, args.time_step, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_target_point=args.use_target_point, NUM_BOX=args.num_box)

    model = load_weight(model, args.ckpt_path)
    if isinstance(model, nn.DataParallel):
        model = model.module

    count_parameters(model)
    model = model.to(device)

    return model


def read_scenario(args, town=["10", "A6", "B3"]):

    test_set = []
    tracking_list = {}
    tp_dict = {}
    
    data_types = ['interactive']
    # skip_list = json.load(open("./skip_scenario.json"))
    skip_list = []

    for data_type in data_types:
        _type_path = os.path.join(args.data_root, data_type)
        tp_dict[data_type] = json.load(open(
            f"./tp_prediction/{data_type}.json"))

        for basic in sorted(os.listdir(_type_path)):
            basic_path = os.path.join(_type_path, basic, 'variant_scenario')

            for variant in os.listdir(basic_path):
                if [data_type, basic, variant] in skip_list:
                    continue
                variant_path = os.path.join(basic_path, variant)

                if basic[:2] in town:
                    
                    tracking_results = np.load(
                        os.path.join(variant_path, "tracking.npy"))
                    tracking_list[basic+'_'+variant] = tracking_results
                    img_path = os.path.join(variant_path, 'rgb/front')
                    
                    for frame in sorted(os.listdir(img_path)):
                        frame_no = frame.split('.')[0]
                        if frame_no in tp_dict[data_type][basic+'_'+variant]:
                            test_set.append([data_type, basic, variant, int(frame_no)])

    return test_set, tracking_list, tp_dict


def normalize_box(trackers):
    """
        return normalized_trackers TxNx4 ndarray:
        [BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_BOTRIGHT_X, BBOX_BOTRIGHT_Y]
    """
    scale_w = args.img_resize[1]/args.raw_img_size[1]
    scale_h = args.img_resize[0]/args.raw_img_size[0]

    normalized_trackers = trackers.copy()
    normalized_trackers[:, :,
                        0] = normalized_trackers[:, :, 0] * scale_w
    normalized_trackers[:, :,
                        2] = normalized_trackers[:, :, 2] * scale_w
    normalized_trackers[:, :,
                        1] = normalized_trackers[:, :, 1] * scale_h
    normalized_trackers[:, :,
                        3] = normalized_trackers[:, :, 3] * scale_h

    return normalized_trackers


def process_tracking(args, tracking_results, frame_no):
    """
        tracking_results Kx10 ndarray:
        [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
        e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
    """

    start, end = frame_no-args.time_step+1, frame_no
    height, width = args.raw_img_size

    t_array = tracking_results[:, 0]
    tracking_index = tracking_results[np.where(t_array == end)[0], 1]
    trackers = np.zeros([args.time_step, args.num_box, 4]).astype(np.float32) # TxNx4

    for t in range(start, end+1):
        current_tracking = tracking_results[np.where(t_array == t)[0]]

        for i, object_id in enumerate(tracking_index):
            current_actor_id_idx = np.where(
                current_tracking[:, 1] == object_id)[0]

            if len(current_actor_id_idx) != 0:
                # x1, y1, x2, y2
                bbox = current_tracking[current_actor_id_idx, 2:6]
                bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
                bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, width)
                bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
                bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, height)
                trackers[t-start, i, :] = bbox

    trackers = normalize_box(trackers)

    return trackers, tracking_index


def test(args, model, sample_list, tracking_list, tp_dict, device):

    time_step = args.time_step
    num_box = args.num_box
    img_resize = args.img_resize

    camera_transforms = transforms.Compose([
        # transforms.Resize(image_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    roi_dict = OrderedDict()
    total_time = 0

    for idx, test_sample in enumerate(sample_list[::], 1):

        data_type, basic, variant, frame_no = test_sample
        variant_path = os.path.join(args.data_root, data_type, basic, "variant_scenario", variant)
        if not basic+'_'+variant in roi_dict:
            roi_dict[basic+'_'+variant] = OrderedDict()

        # get objects tracklet
        trackers, tracking_id = process_tracking(args, tracking_list[basic+'_'+variant], frame_no)
        trackers = to_device(torch.from_numpy(trackers), device)

        # get ego target point
        target_point = np.array(tp_dict[data_type][basic+'_'+variant][f"{frame_no:08d}"], dtype=np.float32)
        target_point = to_device(torch.from_numpy(target_point), device)

        camera_inputs = []
        action_logits = [[0.0, 1.0]]    # dummy logit

       
        """ without intervention """
        # initialize LSTM
        hx = torch.zeros((num_box+1, model.hidden_size)).to(device)
        cx = torch.zeros((num_box+1, model.hidden_size)).to(device)

        start_time = time.time()

        for t in range(0, time_step):

            camera_name = f"{frame_no-time_step+1+t:08d}.jpg"
            camera_path = os.path.join(variant_path, "rgb/front", camera_name)
            img = Image.open(camera_path).convert('RGB')

            camera_input = camera_transforms(img)
            camera_input = to_device(camera_input, device)

            # save for later usage in intervention
            camera_inputs.append(camera_input.clone().detach())
            mask = torch.ones((1, 3, img_resize[0], img_resize[1])).to(device)

            """ ego feature """
            if args.partial_conv:
                ego_feature = model.backbone.features(camera_input, mask)
            else:
                ego_feature = model.backbone.features(camera_input)

            # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
            ego_feature = model.camera_features(ego_feature).reshape(1, 1, -1)

            """ object feature """
            # 1xTxNx4 -> 1xNx4
            tracker = trackers[:, t].reshape(-1, num_box, 4)

            # 1xNx512
            _, obj_feature = model.object_backbone(camera_input, tracker)

            # 1x(1+N)x512
            feature_input = torch.cat((ego_feature, obj_feature), 1)

            if args.use_target_point:

                # 1x2  -> 1x(1+N)x128
                state_feature = model.target_model(target_point)
                # 1x(1+N)x512 -> 1x(1+N)x(512+128)
                feature_input = torch.cat(
                    (feature_input, state_feature), -1)

            # 1x(1+N)x512 -> (1+N)x512
            feature_input = feature_input.reshape(-1, model.fusion_size)

            # LSTM
            hx, cx = model.step(feature_input, hx, cx)

        updated_feature, attn_weights = model.message_passing(hx, trackers, device)

        vel = model.vel_classifier(model.drop(updated_feature))
        vel = model.sigmoid(vel).reshape(-1)
        confidence_go = 1-vel.to('cpu').numpy()[0]

        """ with intervention """
        for i in range(len(tracking_id)):

            # initialize LSTM
            hx = torch.zeros((num_box+1, model.hidden_size)).to(device)
            cx = torch.zeros((num_box+1, model.hidden_size)).to(device)

            for t in range(0, time_step):

                camera_input = camera_inputs[t].clone().detach()

                y1 = int(trackers[0, t, i, 1])  # TOPLEFT_Y
                y2 = int(trackers[0, t, i, 3])  # BOTRIGHT_Y
                x1 = int(trackers[0, t, i, 0])  # TOPLEFT_X
                x2 = int(trackers[0, t, i, 2])  # BOTRIGHT_X

                camera_input[:, :, y1:y2, x1:x2] = 0
                mask = torch.ones((1, 3, img_resize[0], img_resize[1])).to(device)
                mask[:, :, y1:y2, x1:x2] = 0

                """ ego feature """
                if args.partial_conv:
                    ego_feature = model.backbone.features(
                        camera_input, mask)
                else:
                    ego_feature = model.backbone.features(camera_input)

                # 1x2048x8x20 -> 1x512x1x1 ->  1x1x512
                ego_feature = model.camera_features(ego_feature).reshape(1, 1, -1)

                """ object feature """
                # 1xTxNx4 -> 1xNx4
                tracker = trackers[:, t].reshape(-1, num_box, 4).clone().detach()
                tracker[:, i, :] = 0

                # 1xNx512
                _, obj_feature = model.object_backbone(camera_input, tracker)

                # 1x(1+N)x512
                feature_input = torch.cat((ego_feature, obj_feature), 1)

                if args.use_target_point:


                    # 1x2  -> 1x(1+N)x128
                    state_feature = model.target_model(target_point)
                    # 1x(1+N)x512 -> 1x(1+N)x(512+128)
                    feature_input = torch.cat((feature_input, state_feature), -1)

                # 1x(1+N)x512 -> (1+N)x512
                feature_input = feature_input.reshape(-1, model.fusion_size)

                # LSTM
                hx, cx = model.step(feature_input, hx, cx)

            intervened_trackers = torch.ones((1, time_step, num_box, 4)).to(device)
            intervened_trackers[:, :, i, :] = 0.0
            intervened_trackers = intervened_trackers * trackers

            updated_feature, _ = model.message_passing(hx, intervened_trackers, device)

            vel = model.vel_classifier(model.drop(updated_feature))
            vel = model.sigmoid(vel).reshape(-1)

            # score go and score stop
            s_stop = vel.to('cpu').numpy()[0]
            s_go = 1 - s_stop
            action_logits.append([s_go, s_stop])

        end_time = time.time()

        if args.verbose:
            ellip = end_time-start_time
            print(f"{idx}/{len(sample_list)} {frame_no-time_step+1} to {frame_no} Scenario s_go: {float(confidence_go):4f}\ttime:{ellip:.4f}")
            total_time += ellip

        sample_roi_dict = {}
        for actor_id, score, attn in zip(tracking_id, action_logits[1:len(tracking_id)+1], attn_weights[1:len(tracking_id)+1]):
            sample_roi_dict[str(actor_id)] = [score[0], attn.detach().to('cpu').numpy().item()]
        sample_roi_dict["scenario_go"] = np.float64(confidence_go)

        roi_dict[basic+'_'+variant][str(frame_no)] = sample_roi_dict

    print(f"{len(sample_list)} samples in {total_time:.4f} secs")

    return roi_dict


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=f"RiskBench_Dataset", type=str)
    parser.add_argument('--box2d_root', default=f"RiskBench_Dataset", type=str)
    parser.add_argument('--method', choices=["vision"], type=str, required=True)
    parser.add_argument('--phases', default=['test'], type=list)
    parser.add_argument('--ckpt_path', default="", type=str)

    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--epochs', default=1, type=int)

    parser.add_argument('--time_step', default=5, type=int)
    parser.add_argument('--num_box', default=25, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_target_point', action='store_true', default=False)
    parser.add_argument('--raw_img_size', default=[256,640], type=list)
    parser.add_argument('--img_resize', default=[256,640], type=list)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save_roi', action='store_true', default=False)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = create_model(args, device)
    model.train(False)

    sample_list, tracking_list, tp_dict = read_scenario(args)

    with torch.no_grad():
        roi_dict = test(args, model, sample_list, tracking_list, tp_dict, device)

    if args.save_roi:

        ckpt_name = args.ckpt_path.split('/')[-2]
        epoch_name = args.ckpt_path.split('/')[-1].split('-')[1].split('.')[0]
        save_path = f"./ROI/{args.method}-{ckpt_name}_{epoch_name}.json"
        
        with open(save_path, "w") as f:
            json.dump(roi_dict, f, indent=4)
