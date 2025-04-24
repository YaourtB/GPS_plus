from torch.utils.data import Dataset
import argparse

import numpy as np
import os
import random
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from lib.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm

def pts2depth_batch(ptsmap, extrinsic, intrinsic):
    B, S, S, _ = ptsmap.shape
    pts = ptsmap.view(B, -1, 3).permute(0,2,1)
    calib = intrinsic @ extrinsic
    pts = calib[:, :3, :3] @ pts
    pts = pts + calib[:, :3, 3:4]
    pts[:, :2, :] /= (pts[:, 2:, :] + 1e-8)
    depth = 1.0 / (pts[:, 2, :].view(B, S, S) + 1e-8)
    return depth.unsqueeze(1)

def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    H, W = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, device=depth.device), torch.linspace(0.5, W-0.5, W, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(H, W, 3)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    # 
    offset0 = new_intr1[0, 2] - new_intr0[0, 2]  # x-diff of principal points, ref - main
    disparity0 = -new_depth0 * Tf_x  # Tf_x = P[0, 3] = focal_length * baseline
    flow0 = offset0 - disparity0

    # 
    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    # 
    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1

def stereo_depth2flow(rectify0, rectify1, Tf_x):
    new_intr0, new_depth0 = rectify0
    new_intr1, new_depth1 = rectify1


    offset0 = new_intr1[0, 2] - new_intr0[0, 2]  # x-diff of principal points, ref - main
    disparity0 = -new_depth0 * Tf_x  # Tf_x = P[0, 3] = focal_length * baseline
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1

def read_img(name):
    img = np.array(Image.open(name))
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


class StereoHumanDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.use_local_data = opt.use_local_data
        self.phase = phase
        if self.phase == 'train':
            self.data_root = opt.train_data_root
        elif self.phase == 'val':
            self.data_root = opt.val_data_root
        else:
            self.data_root = os.path.join(opt.local_data_root, 'test', self.phase)

        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        #self.img_orig_path = os.path.join(self.data_root, 'img/%s/%02d_orig.jpg')
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%d.png')
        self.depth_init_path = os.path.join(self.data_root, 'depth_init/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parameter/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parameter/%s/%d_extrinsic.npy')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))

        if self.use_local_data:
            if (self.phase == 'train') or (self.phase == 'val'):
                self.local_data_root = os.path.join(opt.local_data_root, self.phase)
            else:
                self.local_data_root = os.path.join(opt.local_data_root, 'test', self.phase)
            
            self.local_img_path = os.path.join(self.local_data_root, 'img/%s/%d.jpg')
            self.local_mask_path = os.path.join(self.local_data_root, 'mask/%s/%d.jpg')
            self.local_parm_path = os.path.join(self.local_data_root, 'parameter/%s/%d_%d.json')

            if os.path.exists(self.local_data_root):
                assert len(os.listdir(os.path.join(self.local_data_root, 'img'))) == len(self.sample_list)
                logging.info(f"Using local data in {self.local_data_root} ...")


    def load_local_stereo_data(self, sample_name, require_pts=True):
        img0_name = self.local_img_path % (sample_name, self.opt.source_id[0])
        img1_name = self.local_img_path % (sample_name, self.opt.source_id[1])
        
        mask0_name = self.local_mask_path % (sample_name, self.opt.source_id[0])
        mask1_name = self.local_mask_path % (sample_name, self.opt.source_id[1])
        
        parm_name = self.local_parm_path % (sample_name, self.opt.source_id[0], self.opt.source_id[1])
        img0 = read_img(img0_name)
        stereo_data = {
            'img0': img0,
            'mask0': read_img(mask0_name),
            'img1': read_img(img1_name),
            'mask1': read_img(mask1_name),
            'camera': load_json_to_np(parm_name)
        }
        
        H_s, W_s, _ = img0.shape
        
        depth_init0 = self.opt.inverse_depth_init * np.ones((H_s, W_s))
        depth_init1 = self.opt.inverse_depth_init * np.ones((H_s, W_s))
        rect0 = stereo_data['camera']['intr0'], depth_init0
        rect1 = stereo_data['camera']['intr1'], depth_init1
        flow0, flow1 = stereo_depth2flow(rect0, rect1, stereo_data['camera']['Tf_x'])
        if self.opt.use_depth_init:
            stereo_data.update({
                'flow0_init': flow0,
                'flow1_init': flow1 
            })

        return stereo_data

    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True, require_pts=True, require_init=True):

        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        depth_name = self.depth_path % (sample_name, source_id)
        depth_init_name = self.depth_init_path % (sample_name, source_id) if self.opt.use_depth_init else None
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)

        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts, pts_init = None, None, None
        if hr_img:
            img = read_img(image_hr_name)
            # img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            intr[:2] *= 2
        else:
            img = read_img(img_name)
        if require_mask:
            mask = read_img(mask_name)
        if require_pts and os.path.exists(depth_name):
            depth = read_depth(depth_name)
            pts = depth2pts(torch.FloatTensor(depth), torch.FloatTensor(extr), torch.FloatTensor(intr))
        if require_init : 
            H_s, W_s, _  = img.shape
            
            depth_init = self.opt.inverse_depth_init * np.ones((H_s, W_s))
            pts_init = depth2pts(torch.FloatTensor(depth_init), torch.FloatTensor(extr), torch.FloatTensor(intr))

        return img, mask, intr, extr, pts, pts_init

    def get_novel_view_tensor(self, sample_name, view_id):
        img, _, intr, extr, _, _ = self.load_single_view(sample_name, view_id, hr_img=self.opt.use_hr_img,
                                                         require_mask=False, require_pts=False, require_init=False)
        height, width = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, fovX=FovX, fovY=FovY,
                                                K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'extr': torch.FloatTensor(extr),  
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center,
            'sample_name': sample_name
        }

        return novel_view_data


    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0

            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'Tf_x': torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'Tf_x': -torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        if 'flow0' in stereo_data:
            flow_tensor, valid_tensor = [], []
            for (flow_view, valid_view) in [('flow0', 'valid0'), ('flow1', 'valid1')]:
                flow = torch.from_numpy(stereo_data[flow_view]).float()
                flow = torch.unsqueeze(flow, dim=0)
                flow_tensor.append(flow)

                valid = torch.from_numpy(stereo_data[valid_view])
                valid = torch.unsqueeze(valid, dim=0)
                valid = valid / 255.0
                valid_tensor.append(valid)

            lmain_data['flow'], lmain_data['valid'] = flow_tensor[0], valid_tensor[0]
            rmain_data['flow'], rmain_data['valid'] = flow_tensor[1], valid_tensor[1]

        if 'flow0_init' in stereo_data:
            flow_init_tensor = []
            for flow_init_view in ['flow0_init', 'flow1_init']:
                flow_init = torch.from_numpy(stereo_data[flow_init_view]).float()
                flow_init = torch.unsqueeze(flow_init, dim=0)
                flow_init_tensor.append(flow_init)

            lmain_data['flow_init'] = flow_init_tensor[0]
            rmain_data['flow_init'] = flow_init_tensor[1]

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if self.use_local_data:
            stereo_np = self.load_local_stereo_data(sample_name, require_pts=True)
        else:
            pass 

        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        if novel_id:
            novel_id = np.random.choice(novel_id)
            dict_tensor.update({
                'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
            })

        return dict_tensor


    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val':
            return self.get_item(index, novel_id=self.opt.val_novel_id)
        else:
            return self.get_item(index, novel_id=self.opt.val_novel_id)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 200
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list) * self.val_boost


if __name__ == '__main__':
    from config.stereo_human_config import ConfigStereoHuman as config
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import kornia.augmentation as K

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, default='/media/ssd/Datasets/real_data/zzr_tmp')
    parser.add_argument('--use_local_data', type=bool, default=False)
    arg = parser.parse_args()

    cfg = config()
    cfg.load("../config/stage2.yaml")
    cfg = cfg.get_cfg()
    cfg.defrost()
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_local_data = arg.use_local_data
    cfg.freeze()

    dataset = StereoHumanDataset(cfg.dataset, phase='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    train_iterator = iter(dataloader)
    for _ in tqdm(range(200000)):
        try:
            data = next(train_iterator)
        except:
            train_iterator = iter(dataloader)
            data = next(train_iterator)

        print(data['novel_view']['img'].shape)

        # data_blob = data['img0'], data['img1'], data['flow'], data['valid']
        # image1, image2, flow, valid = [x.cuda() for x in data_blob]
