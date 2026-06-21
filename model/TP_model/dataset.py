import json
import os
import cv2
import numpy as np
import torch


class RiskBenchDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_root, phase='train', time_step=5, use_gt=False):
        super(RiskBenchDataset, self).__init__()

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.img_root = img_root
        self.time_step = time_step
        self.use_gt = use_gt
        self.data_types = ["interactive", "non-interactive", "obstacle", "collision"][:1]
        
        self.VIEW_MASK = cv2.imread("mask_120degree.png")[:,:,0].copy()/255
        self.target_points = {}
        self.img_list = []

        # skip_list = json.load(open("../../utils/skip_scenario.json"))
        skip_list = []
        
        for data_type in self.data_types:
            self.target_points[data_type] = json.load(open(f"metadata/RiskBench/target_points/target_point_{data_type}.json"))

            type_path = os.path.join(img_root, data_type)

            for basic in sorted(os.listdir(type_path)):
                if not basic[:2] in town:
                    continue
                basic_path = os.path.join(type_path, basic, 'variant_scenario')

                for variant in os.listdir(basic_path):
                    if [data_type, basic, variant] in skip_list:
                        continue
                    seg_folder = os.path.join(basic_path, variant, "bev-seg")

                    if phase == 'test':
                        seg_list = sorted(os.listdir(seg_folder))[self.time_step-1::]
                    else:
                        seg_list = sorted(os.listdir(seg_folder))[self.time_step-1:-20:self.time_step]

                    for frame in seg_list:
                        self.img_list.append([data_type, basic, variant, frame])

        self.img_list = self.img_list[:]


    def onehot_seg(self, bev_seg, N_CLASSES=6):

        """
            # AGENT = 6 (EGO)
            OBSTACLES = 5
            PEDESTRIANS = 4
            VEHICLES = 3
            ROAD_LINE = 2
            ROAD = 1
            UNLABELES = 0
        """

        bev_seg = np.where((bev_seg<6) & (bev_seg>0), bev_seg, 0)
        bev_seg = torch.LongTensor(bev_seg)
        one_hot = torch.nn.functional.one_hot(bev_seg, N_CLASSES).permute(2, 0, 1).float()

        return one_hot[1:]


    def __getitem__(self, index):
        
        data_type, basic, variant, frame = self.img_list[index]
        variant_path = os.path.join(self.img_root, data_type, basic, "variant_scenario", variant)

        gt_seg_list = []
        
        start_frame = int(frame.split('.')[0])
        for frame_id in range(start_frame-self.time_step+1, start_frame+1):

            if self.use_gt:
                seg_path = os.path.join(variant_path, 'bev-seg', f"{frame_id:08d}.npy")
                gt_seg = (np.load(seg_path)[:100]*self.VIEW_MASK)
                new_gt_seg = self.onehot_seg(gt_seg)

            else:
                seg_path = os.path.join(variant_path, 'cvt_bev-seg', f"{frame_id:08d}.npy")
                gt_seg = (np.load(seg_path)*self.VIEW_MASK[None,:,:])
                new_gt_seg = torch.from_numpy(gt_seg)

            gt_seg_list.append(new_gt_seg)

        gt_seg_list = torch.stack(gt_seg_list)

        x, y = self.target_points[data_type][basic+'_'+variant][frame.split('.')[0]]
        x = min(max(-90, x), 90)
        y = min(max(10, y), 90)
        target_point = torch.Tensor([x, y])

        return gt_seg_list, target_point, f"{data_type}#{basic}#{variant}#{start_frame}"

    def __len__(self):
        return len(self.img_list)
