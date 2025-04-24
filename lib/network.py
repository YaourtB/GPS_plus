
import torch
import torch.nn.functional as F
from torch import nn
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from lib.embedder import get_embedder
from lib.attention_module import LocalFeatureTransformer
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.use_flow_init = self.cfg.dataset.use_depth_init
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        
        self.loftr_coarse = LocalFeatureTransformer()
        
        self.raft_stereo = RAFTStereoHuman(self.cfg.raft)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

        flow_init = torch.cat([data['lmain']['flow_init'], data['rmain']['flow_init']], dim=0) if self.use_flow_init else None

        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)

        (feat_c0, feat_c1) = img_feat[2].split(bs)
        
        mask_c0 = mask_c1 = None  

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_cs = torch.cat((feat_c0, feat_c1), 0)
        
        #img_feat[2] = feat_cs 
        img_feat = img_feat[0], img_feat[1], feat_cs
        
        if is_train:
            flow_predictions = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.train_iters)
            # flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)
            flow_loss = None
            metrics = {}
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

        else:
            flow_up = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # regress gaussian parms
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        l_resdepth, r_resdepth =  torch.split(depth_maps, [bs, bs])
        # depth input

        data['lmain']['depth'] += l_resdepth
        data['rmain']['depth'] += r_resdepth

        cut_m = 0
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)  # [B, S*S, 3]
    
            valid = data[view]['mask'][:,:1,:,:] > 0.5  # [B, 1, S, S]
            data[view]['pts_valid'] = valid.view(bs, -1)  # [B, S*S]



        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data

