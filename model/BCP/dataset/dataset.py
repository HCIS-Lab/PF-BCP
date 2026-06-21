import os
import cv2
import random
import json
import time
import numpy as np
import PIL.Image as Image
import torch
import torch.utils.data as data

class VisionDataLayer(data.Dataset):

    def __init__(self, data_root, meta_root, num_box=25, img_resize=(256, 640), raw_img_size=(256, 640),
                 camera_transforms=None, data_type="interactive", time_step=5, phase="train"):

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.data_root = data_root
        self.time_step = time_step
        self.num_box = num_box
        self.phase = phase
        self.camera_transforms = camera_transforms
        self.raw_img_size = raw_img_size
        self.img_resize = img_resize
        self.scale_w = img_resize[1]/raw_img_size[1]
        self.scale_h = img_resize[0]/raw_img_size[0]
        self.data_type = data_type

        self.tp_dict = json.load(open(os.path.join(meta_root, "target_points", f"target_point_{data_type}.json")))
        self.behavior_dict = json.load(open(os.path.join(meta_root, "behavior", f"{data_type}.json")))
        
        self.cnt_labels = np.zeros(2, dtype=np.int32)
        self.inputs = []


        start_time = time.time()

        type_path = os.path.join(data_root, data_type)
        for basic in sorted(os.listdir(type_path)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(type_path, basic, 'variant_scenario')

            for variant in os.listdir(basic_path):
                
                img_folder = os.path.join(basic_path, variant, "rgb/front")
                N = len(os.listdir(img_folder))
                frames, labels = self.get_behavior(data_type, basic, variant, N)

                for frame_no, label in list(zip(frames, labels))[self.time_step-1:-20:1]:
                    self.inputs.append([data_type, basic, variant, frame_no, np.array(label, dtype=np.float32)])
                    self.cnt_labels[int(label)] += 1

        end_time = time.time()

        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def get_behavior(self, data_type, basic, variant, N, start_frame=1):

        first_frame_id = start_frame + self.time_step - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels

    def normalize_box(self, trackers):
        """
            return normalized_trackers TxNx4 ndarray:
            [BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_BOTRIGHT_X, BBOX_BOTRIGHT_Y]
        """

        normalized_trackers = trackers.copy()
        normalized_trackers[:, :,
                            0] = normalized_trackers[:, :, 0] * self.scale_w
        normalized_trackers[:, :,
                            2] = normalized_trackers[:, :, 2] * self.scale_w
        normalized_trackers[:, :,
                            1] = normalized_trackers[:, :, 1] * self.scale_h
        normalized_trackers[:, :,
                            3] = normalized_trackers[:, :, 3] * self.scale_h

        return normalized_trackers

    def process_tracking(self, variant_path, start, end):
        """
            tracking_results Kx10 ndarray:
            [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
            e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
        """

        tracking_results = np.load(
            os.path.join(variant_path, 'tracking.npy'))
        assert len(tracking_results) > 0, f"{variant_path} No tracklet"

        height, width = self.raw_img_size

        t_array = tracking_results[:, 0]
        tracking_index = tracking_results[np.where(t_array == end-1)[0], 1]
        trackers = np.zeros([self.time_step, self.num_box, 4]).astype(np.float32) # TxNx4

        for t in range(start, end):
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

        trackers = self.normalize_box(trackers)

        return trackers, tracking_index


    def __getitem__(self, index):

        data_type, basic, variant, frame_no, label,  = self.inputs[index]
        camera_inputs = []

        for frame_id in range(frame_no-self.time_step+1, frame_no+1):        
            camera_name = f"{int(frame_id):08d}.jpg"
            camera_path = os.path.join(self.data_root, data_type, basic, "variant_scenario", variant, "rgb/front", camera_name)
            img = self.camera_transforms(Image.open(camera_path).convert('RGB').copy())
            camera_inputs.append(img)

        camera_inputs = torch.stack(camera_inputs)

        variant_path = os.path.join(self.data_root, data_type, basic, "variant_scenario", variant)
        trackers, tracking_id = self.process_tracking(variant_path, start=frame_no-self.time_step+1, end=frame_no+1)
        mask = torch.ones((self.time_step, 3, self.img_resize[0], self.img_resize[1]))

        target_point = self.tp_dict[data_type][basic+'_'+variant][f"{frame_no:08d}"]
        target_point = torch.Tensor(target_point)

        return camera_inputs, target_point, trackers, mask, label
    

    def __len__(self):
        return len(self.inputs)


class BEV_SEGDataLayer(data.Dataset):
    
    def __init__(self, data_root, meta_root, num_box=25,\
                  img_resize=(100, 200), data_type="interactive", use_gt=False, time_step=5, phase="train"):

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.data_root = data_root
        self.num_box = num_box
        self.img_resize = img_resize
        self.use_gt = use_gt
        self.time_step = time_step
        self.data_type = data_type

        if self.use_gt:
            self.tp_dict = json.load(open(os.path.join(meta_root, "target_points", f"target_point_{data_type}.json")))
        else:
            self.tp_dict = f"./tp_prediction/interactive.json"
        self.VIEW_MASK = (cv2.imread(os.path.join(meta_root, "mask", "mask_120degree.png"))[:,:,0] != 0).astype(np.float32)
        self.behavior_dict = json.load(open(os.path.join(meta_root, "behavior", f"{data_type}.json")))

        self.cnt_labels = np.zeros(2, dtype=np.int32)
        self.inputs = []


        start_time = time.time()    

        type_path = os.path.join(data_root, data_type)
        for basic in sorted(os.listdir(type_path)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(type_path, basic, 'variant_scenario')

            for variant in os.listdir(basic_path):

                seg_folder = os.path.join(basic_path, variant, "bev-seg")
                N = len(os.listdir(seg_folder))
                frames, labels = self.get_behavior(data_type, basic, variant, N)

                for frame_no, label in list(zip(frames, labels))[self.time_step-1:-20:self.time_step]:
                    self.inputs.append([data_type, basic, variant, frame_no, np.array(label, dtype=np.float32)])
                    self.cnt_labels[int(label)] += 1

        end_time = time.time()

        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def get_behavior(self, data_type, basic, variant, N, start_frame=1):

        first_frame_id = start_frame + self.time_step - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels


    def onehot_seg(self, bev_seg, N_CLASSES=5):
        
        """
            src:
                AGENT = 6
                OBSTACLES = 5
                PEDESTRIANS = 4
                VEHICLES = 3
                ROAD_LINE = 2
                ROAD = 1
                UNLABELES = 0
            new (return):
                OBSTACLES = 4
                PEDESTRIANS = 3
                VEHICLES = 2
                ROAD_LINE = 1
                ROAD = 0
        """

        if self.use_gt:
            new_bev_seg = np.where((bev_seg<6) & (bev_seg>0), bev_seg, 0)
            new_bev_seg = torch.LongTensor(new_bev_seg)
            one_hot = torch.nn.functional.one_hot(new_bev_seg, N_CLASSES+1).permute(2, 0, 1).float()

            return one_hot[1:]
        else:
            return torch.from_numpy(bev_seg)


    def __getitem__(self, index):

        data_type, basic, variant, frame, label = self.inputs[index]
        variant_path = os.path.join(self.data_root, data_type, basic, "variant_scenario", variant)
        
        gt_seg_list = []

        for frame_id in range(frame-self.time_step+1, frame+1):

            if self.use_gt:
                seg_path = os.path.join(variant_path, "bev-seg", f"{frame_id:08d}.npy")
                gt_seg = (np.load(seg_path)[:100]*self.VIEW_MASK)

            else:
                seg_path = os.path.join(variant_path, "cvt_bev-seg", f"{frame_id:08d}.npy")
                gt_seg = (np.load(seg_path)[:100]*self.VIEW_MASK)

            # new_gt_seg : Cx100x200
            new_gt_seg = self.onehot_seg(gt_seg)
            gt_seg_list.append(new_gt_seg)

        gt_seg_list = torch.stack(gt_seg_list)

        # padding tracker (dummy)
        trackers = np.zeros([self.time_step, self.num_box, 4]).astype(np.float32)
        mask = torch.ones((self.time_step, 4, self.img_resize[0], self.img_resize[1]))

        target_point = self.tp_dict[basic+'_'+variant][f"{frame:08d}"]
        target_point = torch.Tensor(target_point)

        return gt_seg_list, target_point, trackers, mask, label


    def __len__(self):
        return len(self.inputs)


class PFDataLayer(data.Dataset):
    
    def __init__(self, data_root, meta_root, num_box=20,\
                  img_resize=(100, 200), data_type="interactive", use_gt=False, time_step=5, phase="train"):

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.data_root = data_root
        self.num_box = num_box
        self.img_resize = img_resize
        self.use_gt = use_gt
        self.time_step = time_step
        self.data_type = data_type


        if self.use_gt:
            self.tp_dict = json.load(open(os.path.join(meta_root, "target_points", f"target_point_{data_type}.json")))
        else:
            self.tp_dict = f"./tp_prediction/interactive.json"
        self.VIEW_MASK = (cv2.imread(os.path.join(meta_root, "mask", "mask_120degree.png"))[:,:,0] != 0).astype(np.float32)
        self.behavior_dict = json.load(open(os.path.join(meta_root, "behavior", f"{data_type}.json")))

        self.cnt_labels = np.zeros(2, dtype=np.int32)
        self.inputs = []


        start_time = time.time()

        type_path = os.path.join(self.data_root, self.data_type)
        for basic in sorted(os.listdir(type_path)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(type_path, basic, 'variant_scenario')

            for variant in os.listdir(basic_path):

                seg_folder = os.path.join(basic_path, variant, "bev-seg")
                N = len(os.listdir(seg_folder))
                frames, labels = self.get_behavior(self.data_type, basic, variant, N)

                if phase == 'test':
                    frame_list = list(zip(frames, labels))[::]
                else:
                    frame_list = list(zip(frames, labels))[:-20:1]
                
                for frame_no, label in frame_list:
                    self.inputs.append([self.data_type, basic, variant, frame_no, np.array(label, dtype=np.float32)])
                    self.cnt_labels[int(label)] += 1

        end_time = time.time()

        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def get_behavior(self, data_type, basic, variant, N, start_frame=1):

        first_frame_id = start_frame + self.time_step - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels


    def __getitem__(self, index):

        data_type, basic, variant, frame, label = self.inputs[index]
        variant_path = os.path.join(self.data_root, data_type, basic, "variant_scenario", variant)
        gt_pf_list = []

        for frame_id in range(frame-self.time_step+1, frame+1):

            if self.use_gt:
                pf_path = os.path.join(variant_path, "actor_pf_npy", f"{frame_id:08d}.npy")
            else:
                pf_path = os.path.join(variant_path, "pre_cvt_actor_pf_npy", f"{frame_id:08d}.npy")

            npy_file = np.load(pf_path, allow_pickle=True).item()
            
            gt_pf = np.zeros((3, 100, 200), dtype=np.float32)
            gt_pf[0] = npy_file['all_actor']
            gt_pf[1] = npy_file['roadline']
            gt_pf[2] = npy_file['attractive']

            gt_pf = gt_pf*self.VIEW_MASK[None, ...]
            gt_pf_list.append(torch.from_numpy(gt_pf))

        gt_pf_list = torch.stack(gt_pf_list)

        # padding tracker (dummy)
        trackers = np.zeros([self.time_step, self.num_box, 4]).astype(np.float32)
        mask = torch.ones((self.time_step, 4, self.img_resize[0], self.img_resize[1]))

        target_point = self.tp_dict[basic+'_'+variant][f"{frame:08d}"]
        target_point = torch.Tensor(target_point)

        return gt_pf_list, target_point, trackers, mask, label


    def __len__(self):
        return len(self.inputs)

