import torch
import torch.nn as nn
from .backbone import Riskbench_backbone
from .pdresnet50 import pdresnet50
import torch.nn.functional as F

__all__ = [
    'GCN',
]


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)



class GCN(nn.Module):
    def __init__(self, method, time_steps=5, pretrained=True, partialConv=True, use_target_point=False, NUM_BOX=0):
        super(GCN, self).__init__()

        self.method = method
        self.time_steps = time_steps
        self.pretrained = pretrained
        self.partialConv = partialConv
        self.use_target_point = use_target_point
        self.hidden_size = 512

        # build backbones
        if self.partialConv:
            if self.method in ["vision"]:
                channel = 3
                
            elif self.method in ["bev_seg"]:
                channel = 5
                self.pretrained = False

            elif self.method in ["pf"]:
                channel = 3
                self.pretrained = False
            else:
                print("No input channel!")
                exit()

            self.backbone = pdresnet50(pretrained=self.pretrained, channel=channel)

        if self.method in ["vision"]:
            self.num_box = NUM_BOX
            self.object_backbone = Riskbench_backbone(
                roi_align_kernel=8, n=self.num_box, pretrained=pretrained)

            # 2d conv after backbone
            self.camera_features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            )

        else:
        # if False:
            self.num_box = 0        
            self.img_encoder = nn.Sequential(
                nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 16)),
            )
            self.camera_features = nn.Sequential(
                nn.Conv2d(32, 4, kernel_size=1, stride=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.Flatten()
            )

        if self.use_target_point:
            self.state_features = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(inplace=True),
            )

        # temporal modeling
        self.fusion_size = 512 + (128 if self.use_target_point else 0)
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        # gcn module
        self.emb_size = self.hidden_size
        self.fc_emb_1 = nn.Linear(
            self.hidden_size, self.emb_size, bias=False)
        self.fc_emb_2 = nn.Linear(self.emb_size * 2, 1)

        # final classifier
        self.vel_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.sigmoid = nn.Sigmoid()


    def target_model(self, target_point):
        
        batch_size = target_point.shape[0]

        # Bx2 -> Bx128 -> Bx1x128 -> Bx(1+N)x128
        goal_feature = self.state_features(target_point)
        goal_feature = goal_feature.reshape(batch_size, 1, -1)
        goal_feature = goal_feature.repeat(1, self.num_box+1, 1)

        return goal_feature


    def message_passing(self, input_feature, trackers, device=0):
        #############################################
        # input_feature:    (B(1+N))xH
        # trackers:         BxTxNx4
        #############################################
        
        B = len(trackers)

        if self.method in ["vision"]:
            num_box = trackers.shape[2]+1
            mask = torch.ones((B, num_box))
            mask[:, 1:] = trackers[:, -1, :, 2]+trackers[:, -1, :, 3]
            mask = mask != 0  # (B, N, 1)
        else:
            num_box = 1
            mask = torch.ones((B, num_box))

        # (B(1+N))xH -> (B(1+N))xself.emb_size
        emb_feature = self.fc_emb_1(input_feature)
        # (B(1+N))xself.emb_size ->  Bx(1+N)xself.emb_size
        emb_feature = emb_feature.reshape(-1, num_box, self.emb_size)

        # Bx(1+N)xself.emb_size
        ego_feature = emb_feature[:, 0, :].reshape(-1, 1, self.emb_size).repeat(1, num_box, 1)
        # Bx(1+N)x(2*self.emb_size)
        emb_feature = torch.cat((ego_feature, emb_feature), 2)
        # (B(1+N))x(2*self.emb_size)
        emb_feature = emb_feature.reshape(-1, 2 * self.emb_size)

        # Bx(1+N)x1
        emb_feature = self.fc_emb_2(emb_feature).reshape(-1, num_box, 1)
        emb_feature[~(mask.byte().to(torch.bool))] = torch.tensor(
            [-float("Inf")]).to(device)

        # Bx(1+N)x1
        attn_weights = F.softmax(emb_feature, dim=1)
        # (B(1+N))x1
        attn_weights = attn_weights.reshape(-1, 1)

        # BxH
        ori_ego_feature = input_feature.reshape(-1,
                                                num_box, self.hidden_size)[:, 0, :]
        # (B(1+N))xH
        input_feature = input_feature.reshape(-1, self.hidden_size)

        # Bx(1+N)xH
        fusion_feature = (
            input_feature * attn_weights).reshape(-1, num_box, self.hidden_size)
        # BxH
        fusion_feature = torch.sum(fusion_feature, 1)
        # Bx(2*H)
        fusion_feature = torch.cat((ori_ego_feature, fusion_feature), 1)

        return fusion_feature, attn_weights


    def step(self, camera_input, hx, cx):

        fusion_input = camera_input
        hx, cx = self.lstm(self.drop(fusion_input), (hx, cx))

        return hx, cx


    def forward(self, camera_inputs, target_point=None, trackers=None, mask=None, device='cuda'):

        ###########################################
        #  camera_input     :   BxTxCxWxH
        #  tracker          :   BxTxNx4
        #  intention_inputs :   Bx10
        ###########################################

        # Record input size
        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        c = camera_inputs.shape[2]
        h = camera_inputs.shape[3]
        w = camera_inputs.shape[4]

        # initialize LSTM
        hx = torch.zeros(
            (batch_size*(1+self.num_box), self.hidden_size)).to(device)
        cx = torch.zeros(
            (batch_size*(1+self.num_box), self.hidden_size)).to(device)

        """ ego feature """
        # BxTxCxHxW -> (BT)xCxHxW
        camera_inputs = camera_inputs.reshape(-1, c, h, w)

        # (BT)x2048x8x20
        if self.partialConv:
            if self.method in ["vision"]:
                ego_features = self.backbone.features(camera_inputs, mask.reshape(-1, c, h, w))
            else:
                ego_features = self.img_encoder(camera_inputs)

        else:
            ego_features = self.backbone.features(camera_inputs)

        # Reshape the ego_features to LSTM
        c = ego_features.shape[1]
        h = ego_features.shape[2]
        w = ego_features.shape[3]

        # (BT)x2048x8x20 -> BxTx2048x8x20
        ego_features = ego_features.reshape(batch_size, t, c, h, w)

        """ object feature"""
        if self.method in ["vision"]:
            # BxTxNx4 -> (BT)xNx4
            tracker = trackers.reshape(-1, self.num_box, 4)

            # (BT)xNx512
            _, obj_features = self.object_backbone(camera_inputs, tracker)

            # BxTxNx512
            obj_features = obj_features.reshape(batch_size, t, self.num_box, -1)

        # Running LSTM
        for l in range(0, self.time_steps):

            # BxTx2048x8x20 -> Bx2048x8x20
            ego_feature = ego_features[:, l].clone()

            # Bx2048x8x20 -> Bx512x1x1 ->  Bx1x512
            ego_feature = self.camera_features(
                ego_feature).reshape(batch_size, 1, -1)

            if self.method in ["vision"]:
                # BxTxNx512 -> BxNx512
                obj_feature = obj_features[:, l].clone()
                # Bx(1+N)x512
                feature_input = torch.cat((ego_feature, obj_feature), 1)
            else:
                feature_input = ego_feature

            if self.use_target_point:

                # Bx2  -> Bx(1+N)x128
                goal_feature = self.target_model(target_point)
                
                # Bx(1+N)x512 -> Bx(1+N)x(512+128)
                feature_input = torch.cat(
                    (feature_input, goal_feature), -1)

            # Bx(1+N)x512 -> (B(1+N))x512
            feature_input = feature_input.reshape(-1, self.fusion_size)

            # LSTM
            hx, cx = self.step(feature_input, hx, cx)

        updated_feature, _ = self.message_passing(hx, trackers, device)

        vel = self.vel_classifier(self.drop(updated_feature))
        vel = self.sigmoid(vel).reshape(-1)

        return vel
