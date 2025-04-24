
import torch
import trimesh
import numpy as np
import cv2
import kornia


# https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/utils.py
def depth2mesh(pts, depth):
    S = depth.shape[3]

    pts = pts.view(S, S, 3).detach().cpu().numpy()
    diff_depth = depth_filter(depth)
    depth[diff_depth > 0.04] = 0
    valid_depth = depth != 0
    valid_depth = kornia.morphology.erosion(valid_depth.float(), torch.ones(3, 3).cuda())
    valid_depth = valid_depth.squeeze().detach().cpu().numpy()

    f_list = []
    for i in range(S-1):
        for j in range(S-1):
            if valid_depth[i, j] and valid_depth[i+1, j+1]:
                if valid_depth[i+1, j]:
                    f_list.append([int(i*S+j+1), int((i+1)*S+j+1), int((i+1)*S+j+2)])
                if valid_depth[i, j+1]:
                    f_list.append([int(i*S+j+1), int((i+1)*S+j+2), int(i*S+j+2)])

    obj_out = trimesh.Trimesh(vertices=pts.reshape(-1, 3), faces=np.array(f_list))

    return obj_out





def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def flow2depth(data):
    offset = data['ref_intr'][:, 0, 2] - data['intr'][:, 0, 2]
    offset = torch.broadcast_to(offset[:, None, None, None], data['flow_pred'].shape)
    disparity = offset - data['flow_pred']
    depth = -disparity / data['Tf_x'][:, None, None, None]
    depth *= data['mask'][:, :1, :, :]

    return depth


def depth_filter(depth):
    diff_depth = torch.zeros_like(depth)
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., 1:, :] - depth[..., :-1, :])[..., :-1, 1:-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :-1, :] - depth[..., 1:, :])[..., 1:, 1:-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :, 1:] - depth[..., :, :-1])[..., 1:-1, :-1]
    diff_depth[..., 1:-1, 1:-1] += torch.abs(depth[..., :, :-1] - depth[..., :, 1:])[..., 1:-1, 1:]

    return diff_depth


def perspective(pts, calibs):
    # pts: [B, N, 3]
    # calibs: [B, 3, 4]
    pts = pts.permute(0, 2, 1)
    pts = torch.bmm(calibs[:, :3, :3], pts)
    pts = pts + calibs[:, :3, 3:4]
    pts[:, :2, :] /= pts[:, 2:, :]
    pts = pts.permute(0, 2, 1)
    return pts


def flow2render_depth(data, taichi_render, pts_out_path=None):
    B = data['lmain']['intr'].shape[0]
    S = data['lmain']['img_ori'].shape[2]
    data_select = data['lmain']

    depth_pred = flow2depth(data_select).clone()  # [B, 1, 1024, 1024]
    # diff_depth = depth_filter(depth_pred)
    # depth_pred[diff_depth > 0.025] = 0  # 
    valid = depth_pred != 0  # [B, 1, 1024, 1024]

    pts = depth2pc(depth_pred, data_select['extr'], data_select['intr'])  # [B, S*S, 3]
    valid = valid.view(B, -1, 1).squeeze(2)  # [B, S*S]
    pts_valid = torch.zeros_like(pts)
    pts_valid[valid] = pts[valid]

    if pts_out_path:
        # depth mesh
        obj_out = depth2mesh(pts_valid.clone(), depth_pred.clone())
        obj_out.export(pts_out_path + '/%s_depth_rebuild.obj' % (data['name']))

        # color ply
        # pts_show = pts[valid]
        # color_show = data_select['img'].permute(0, 2, 3, 1).view(B, -1, 3)[valid]
        # ply_out = trimesh.points.PointCloud(vertices=pts_show.detach().cpu().numpy(),
        #                                     colors=color_show.detach().cpu().numpy().astype(np.uint8))
        # ply_out.export(pts_out_path + '/%s_lmain.ply' % (data['name']))

    calib_ori = torch.matmul(data_select['intr_ori'], data_select['extr_ori'])
    pts_valid = perspective(pts_valid, calib_ori)
    pts_valid[:, :, 2:] = 1.0 / (pts_valid[:, :, 2:] + 1e-8)

    # render
    taichi_mask = valid.view(B, -1, 1).float()  # [B, S*S, 1]
    render_depth = torch.zeros((B, 1, S, S), device=pts.device).float()
    taichi_render.render_perspective_depth(pts_valid.contiguous(), taichi_mask, render_depth)

    # depth pad
    kernel = torch.ones(3, 3).cuda()
    with torch.no_grad():
        ti_depth_pad = kornia.morphology.dilation(render_depth.clone(), kernel)
        mask_pad = ti_depth_pad != 0
        mask_pad = kornia.morphology.erosion(mask_pad.float(), kernel)
        mask_pad = kornia.morphology.erosion(mask_pad, kernel)
    taichi_render.pad_depth(render_depth, ti_depth_pad, mask_pad)
    data['lmain']['depth_ori'] = render_depth

    return data

def flow2render_depth_my(data, taichi_render, pts_out_path=None, select_view = 'lmain'):
    B = data[select_view]['intr'].shape[0]
    S = data[select_view]['img_ori'].shape[2]
    data_select = data[select_view]

    depth_pred = flow2depth(data_select).clone()  # [B, 1, 1024, 1024]
    # diff_depth = depth_filter(depth_pred)
    # depth_pred[diff_depth > 0.025] = 0  # 
    valid = depth_pred != 0  # [B, 1, 1024, 1024]

    pts = depth2pc(depth_pred, data_select['extr'], data_select['intr'])  # [B, S*S, 3]
    valid = valid.view(B, -1, 1).squeeze(2)  # [B, S*S]
    pts_valid = torch.zeros_like(pts)
    pts_valid[valid] = pts[valid]

    if pts_out_path:
        # depth mesh
        obj_out = depth2mesh(pts_valid.clone(), depth_pred.clone())
        obj_out.export(pts_out_path + '/%s_depth_rebuild.obj' % (data['name']))

        # color ply
        # pts_show = pts[valid]
        # color_show = data_select['img'].permute(0, 2, 3, 1).view(B, -1, 3)[valid]
        # ply_out = trimesh.points.PointCloud(vertices=pts_show.detach().cpu().numpy(),
        #                                     colors=color_show.detach().cpu().numpy().astype(np.uint8))
        # ply_out.export(pts_out_path + '/%s_lmain.ply' % (data['name']))

    calib_ori = torch.matmul(data_select['intr_ori'], data_select['extr_ori'])
    pts_valid = perspective(pts_valid, calib_ori)
    pts_valid[:, :, 2:] = 1.0 / (pts_valid[:, :, 2:] + 1e-8)

    # render
    taichi_mask = valid.view(B, -1, 1).float()  # [B, S*S, 1]
    render_depth = torch.zeros((B, 1, S, S), device=pts.device).float()
    taichi_render.render_perspective_depth(pts_valid.contiguous(), taichi_mask, render_depth)

    # depth pad
    kernel = torch.ones(3, 3).cuda()
    with torch.no_grad():
        ti_depth_pad = kornia.morphology.dilation(render_depth.clone(), kernel)
        mask_pad = ti_depth_pad != 0
        mask_pad = kornia.morphology.erosion(mask_pad.float(), kernel)
        mask_pad = kornia.morphology.erosion(mask_pad, kernel)
    taichi_render.pad_depth(render_depth, ti_depth_pad, mask_pad)
    data[select_view]['depth_ori'] = render_depth

    return data