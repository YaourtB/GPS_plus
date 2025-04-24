from __future__ import print_function, division

import argparse
import logging

import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from lib.human_loader import StereoHumanDataset, load_json_to_np, stereo_depth2flow
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.utils import stereo_rectify
from lib.gs_utils.image_utils import psnr
import lpips
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.human_loader import depth2pts, pts2depth_batch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
import trimesh 
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image

import json 

from copy import deepcopy


def read_calib(calib):  
    # print(calib)
    
    R = np.array(calib['R']).reshape((3, 3))
    T = np.array(calib['T']).reshape((3, 1))
    extr = np.zeros((3, 4))
    extr[:3, :3]  = R
    extr[:3, 3:] = T
    intr = np.zeros((3, 3))
    intr[:3, :3] = np.array(calib['K']).reshape((3, 3))
    H = 2048
    W = 1500
    if W>H:
        intr[0, 2] -= (W - H) / 2
        intr[:2] *= RS / H
    else:
        intr[1, 2] -= (H - W) / 2
        intr[:2] *= RS / W
    calib = intr @ extr
    return extr, intr, calib

def extr_interpolate(RS, cam_id_list, s_id):
    # interpolate novel extr 
    
    novel_extr_list = []
    novel_intr_list = []

    for i in range(1):
        novel_extrs = []
        novel_intrs = []

        calib_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n+'_process', 'calibration_full.json')
        with open(calib_path, 'r') as f:
            calib_full = json.load(f)

        cam = cam_id_list[0]
        extr0, intr0, calib0 = read_calib(calib_full[cam])
        
        cam = cam_id_list[1]
        extr1, intr1, calib1 = read_calib(calib_full[cam])
        
        
        pose_0 = np.eye(4)
        pose_1 = np.eye(4)
        pose_0[:3] = extr0
        pose_1[:3] = extr1
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        for ratio in np.linspace(0., 0.95, LOOP_NUM):
            rot_0 = pose_0[:3, :3]
            rot_1 = pose_1[:3, :3]
            rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
            key_times = [0, 1]
            slerp = Slerp(key_times, rots)
            rot = slerp(ratio)
            pose = np.diag([1.0, 1.0, 1.0, 1.0])
            pose = pose.astype(np.float32)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
            pose = np.linalg.inv(pose)
            novel_extrs.append(pose[:3])
            novel_intrs.append((1.0 - ratio) * intr0 + ratio * intr1)

        
        novel_extr_list = novel_extr_list + [novel_extrs]
        novel_intr_list = novel_intr_list + [novel_intrs]



    return novel_extr_list, novel_intr_list

class StereoHumanModel(nn.Module):
    def __init__(self, cfg, ckpt_path, novel_extrs, novel_intrs, s_id = 1):
        super().__init__()
        
        self.model = RtStereoHumanModel(cfg, with_gs_render=True)# RtStereoHumanModel(cfg, True)
        ckpt = torch.load(ckpt_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        self.model = self.model.cuda()
        self.model.eval()

        self.novel_extrs = novel_extrs
        self.novel_intrs = novel_intrs 
        self.img_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n + '_process', 'img')
        self.msk_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n + '_process', 'mask')

        depth_init0 = cfg.dataset.inverse_depth_init * np.ones((1024, 1024))
        depth_init1 = cfg.dataset.inverse_depth_init * np.ones((1024, 1024))
        #depth_init = torch.FloatTensor(depth_init).cuda()

        parm_name = os.path.join(cfg.dataset.local_data_root, 'test', tar_n+'_process', 'parameter', tar_n+'_s%d_0000' % s_id, '%s_%s.json' % (str(cfg.dataset.source_id[0]), str(cfg.dataset.source_id[1])))

        camera = load_json_to_np(parm_name)
        self.s_id = s_id
        self.Tf_x = np.array([camera['Tf_x']])
        extr0 = camera['extr0']
        extr1 = camera['extr1']
        
        intr0 = camera['intr0']
        intr1 = camera['intr1']

        rect0 = intr0, depth_init0
        rect1 = intr1, depth_init1
        flow0, flow1 = stereo_depth2flow(rect0, rect1, camera['Tf_x'])

        self.lpts_init = torch.FloatTensor(flow0).cuda()
        self.rpts_init = torch.FloatTensor(flow1).cuda()
        
        self.intrinsics = [torch.FloatTensor(intr0), torch.FloatTensor(intr1)] 
        self.extrinsics = [torch.FloatTensor(extr0), torch.FloatTensor(extr1)] 
        
    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def get_item_free(self, frame_id, view_id):
        img0 = np.array(Image.open(self.img_path+'/%s_s%d_%04d/%d.jpg'%(tar_n, self.s_id, frame_id, from_list[0]))).astype(np.float32)
        img1 = np.array(Image.open(self.img_path+'/%s_s%d_%04d/%d.jpg'%(tar_n, self.s_id, frame_id, to_list[0]))).astype(np.float32)
        img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).cuda()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).cuda()
        
        msk0 = np.array(Image.open(self.msk_path+'/%s_s%d_%04d/%d.jpg'%(tar_n, self.s_id, frame_id, from_list[0]))).astype(np.float32)
        msk1 = np.array(Image.open(self.msk_path+'/%s_s%d_%04d/%d.jpg'%(tar_n, self.s_id, frame_id, to_list[0]))).astype(np.float32)
        msk0 = torch.from_numpy(msk0).permute(2, 0, 1).unsqueeze(0).cuda()
        msk1 = torch.from_numpy(msk1).permute(2, 0, 1).unsqueeze(0).cuda()
        

        img0 = 2 * (img0 / 255.0) - 1.0
        img1 = 2 * (img1 / 255.0) - 1.0

        msk0 /= 255
        msk1 /= 255 



        intr0_ = self.intrinsics[0].unsqueeze(0).cuda()
        intr1_ = self.intrinsics[1].unsqueeze(0).cuda()
        extr0 = self.extrinsics[0].unsqueeze(0).cuda()
        extr1 = self.extrinsics[1].unsqueeze(0).cuda()
        Tf_x = torch.FloatTensor([self.Tf_x[0]]).float().cuda().clone()

        flow0, flow1 = self.lpts_init, self.rpts_init

        intr0 = intr0_.clone()
        intr1 = intr1_.clone()

        l_view = {
            'img': img0,
            'mask': msk0,
            'intr': intr0,
            'ref_intr': intr1,
            'extr': extr0,
            'Tf_x': Tf_x, 
            'flow_init': flow0.unsqueeze(0).unsqueeze(0)  
        }

        r_view = {
            'img': img1,
            'mask': msk1,
            'intr': intr1,
            'ref_intr': intr0,
            'extr': extr1,
            'Tf_x': -Tf_x, 
            'flow_init': flow1.unsqueeze(0).unsqueeze(0)  
        }

        novel_intr = self.novel_intrs[0][view_id]#[frame_id] # from-dependent not frame-dependent

        novel_extr = self.novel_extrs[0][view_id]


        width, height = 1024, 1024
        R = np.array(novel_extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(novel_extr[:3, 3], np.float32)

        FovX = focal2fov(novel_intr[0, 0], width)
        FovY = focal2fov(novel_intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY,
                                                K=novel_intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view = {
            'height': [height],
            'width': [width],
            'FovX': [torch.FloatTensor(np.array(FovX)).cuda()],
            'FovY': [torch.FloatTensor(np.array(FovY)).cuda()],
            'world_view_transform':[world_view_transform.cuda()],
            'full_proj_transform': [full_proj_transform.cuda()],
            'camera_center': [camera_center.cuda()]
        }

        dict_tensor = {
            'lmain': l_view,
            'rmain': r_view,
            'novel_view': novel_view
        }

        return dict_tensor

if __name__ == '__main__':
    # python run_interpolation.py -i example_data
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input sequence') # 'zymlqj'
    arg = parser.parse_args()
    
    tar_n = arg.input
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/stage.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = 'gps_plus' # TODO

    cfg.record.show_path = "experiments/%s/show_free_%s" % (cfg.exp_name, tar_n)
    cfg.restore_ckpt = 'ckpt/gps_plus_latest.pth' # TODO
    cfg.freeze()
    LOOP_NUM = 20

    Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)


    from_list = [0]
    to_list = [1]
    RS = 1024
    data_root = cfg.dataset.val_data_root
    
    novel_extrs, novel_intrs = extr_interpolate(RS, ['22139908', '22139909'], 1)
    print(len(novel_extrs[0]))
    
    render = StereoHumanModel(cfg, cfg.restore_ckpt,\
        [novel_extrs[0]], novel_intrs, 1)
    
    novel_extrs, novel_intrs = extr_interpolate(RS, ['22139909', '22139914'], 2)
    print(len(novel_extrs[0]))

    render2 = StereoHumanModel(cfg, cfg.restore_ckpt,\
        [novel_extrs[0]], novel_intrs, 2)
    
    novel_extrs, novel_intrs = extr_interpolate(RS, ['22139914', '22139906'], 3)
    print(len(novel_extrs[0]))

    render3 = StereoHumanModel(cfg, cfg.restore_ckpt,\
        [novel_extrs[0]], novel_intrs, 3)
    
    cut = 100
    tar_ply = -10
    start_frame = 0
    end_frame = 60

    for fr_i in tqdm(range(start_frame, end_frame)):
        wi_ct = fr_i // LOOP_NUM
        scene_id = (wi_ct) % 6
        with torch.no_grad():
            if scene_id == 0:
                data = render.get_item_free(fr_i, fr_i % LOOP_NUM)
                out = render.model(data)
            elif scene_id == 1:
                data = render2.get_item_free(fr_i, fr_i % LOOP_NUM)
                out = render2.model(data)
            elif scene_id == 2:
                data = render3.get_item_free(fr_i, fr_i % LOOP_NUM)
                out = render3.model(data)
            elif scene_id == 3:
                data = render3.get_item_free(fr_i, (LOOP_NUM-1) - (fr_i % LOOP_NUM))
                out = render3.model(data)
            elif scene_id == 4:
                data = render2.get_item_free(fr_i, (LOOP_NUM-1) - (fr_i % LOOP_NUM))
                out = render2.model(data)
            elif scene_id == 5:
                data = render.get_item_free(fr_i, (LOOP_NUM-1) - (fr_i % LOOP_NUM))
                out = render.model(data)
            else:
                exit()
            data = pts2render(data, bg_color=cfg.dataset.bg_color) 
            tmp_novel = data['novel_view']['img_pred'][0].detach()
            tmp_novel *= 255
            tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
            tmp_img_name = '%s/%03d.jpg' % (cfg.record.show_path, fr_i)
            cv2.imwrite(tmp_img_name, tmp_novel[cut:(RS-cut), cut:(RS-cut), ::-1].astype(np.uint8))
