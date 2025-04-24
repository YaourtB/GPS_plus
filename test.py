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

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import warnings
import trimesh 
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        
        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)

        self.val_set = StereoHumanDataset(self.cfg.dataset, phase=cfg.seq_name)
        self.val_loader = DataLoader(self.val_set, batch_size=self.bs, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            print('load good ckpt')
            self.load_ckpt(self.cfg.restore_ckpt)

        self.model.eval()
        self.model.raft_stereo.freeze_bn()  # We keep BatchNorm frozen in Raft-Stereo
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)


    def val(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        for idx in tqdm(range(self.len_val)):
            data = self.fetch_data(phase='val')

            view_id = data['novel_view']['view_id'][0,0].item()
            s_name = data['novel_view']['sample_name']

            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                tmp_novel = data['novel_view']['img_pred'][0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = '%s/%s_%02d.jpg' % (cfg.record.show_path, s_name[0], view_id)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
 

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    # python test_stage2.py -i example_data -v 2
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input sequence') # 'example_data'
    parser.add_argument('-v', '--view', type=int, required=True, help='valid view')      # 3
    arg = parser.parse_args()

    for seq_name in [arg.input+'_process']:
        for views_n in [arg.view]:
            cfg = config()
            cfg.load("config/stage.yaml")
            cfg = cfg.get_cfg()

            cfg.defrost()
            dt = datetime.today()
            cfg.exp_name = 'gps_plus' # TODO
            cfg.record.show_path = "experiments/%s/show_%s" % (cfg.exp_name, seq_name)
            cfg.seq_name = seq_name 
            cfg.dataset.val_novel_id = [views_n] 
            cfg.restore_ckpt = 'ckpt/gps_plus_latest.pth' # TODO

            cfg.freeze()
            print(cfg.restore_ckpt)
            print(cfg.batch_size)
            

            Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)



            trainer = Trainer(cfg)
            trainer.val()
