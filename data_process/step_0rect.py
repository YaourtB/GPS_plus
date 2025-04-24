import numpy as np 
import os
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
from pathlib import Path
import json
import glob 
from tqdm import tqdm 
import argparse 
import shutil 

def get_rectified_stereo_data(main_view_data, ref_view_data, img_sz):
    # define view 0 as main and 1 as reference
    img0, intr0, extr0 = main_view_data
    img1, intr1, extr1 = ref_view_data
    
    W, H = img_sz[0], img_sz[1]
    r0, t0 = extr0[:3, :3], extr0[:3, 3:]
    r1, t1 = extr1[:3, :3], extr1[:3, 3:]
    inv_r0 = r0.T
    inv_t0 = - r0.T @ t0
    E0 = np.eye(4)
    E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
    E1 = np.eye(4)
    E1[:3, :3], E1[:3, 3:] = r1, t1
    E = E1 @ E0
    R, T = E[:3, :3], E[:3, 3]
    dist0, dist1 = np.zeros(4), np.zeros(4)
    move_t = (H-W)/2
    # https://blog.csdn.net/qq_25458977/article/details/114829674
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)

    new_extr0 = R0 @ extr0
    new_intr0 = P0[:3, :3]
    new_extr1 = R1 @ extr1
    new_intr1 = P1[:3, :3]
    Tf_x = np.array(P1[0, 3])
    
    camera = {
        'intr0': new_intr0,
        'intr1': new_intr1,
        'extr0': new_extr0,
        'extr1': new_extr1,
        'Tf_x': Tf_x
    }

    mask0 = np.ones_like(img0)*255
    mask1 = np.ones_like(img1)*255
    rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
    new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)
    new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
    new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
    rectify0 = new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y
    rectify1 = new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y

    stereo_data = {
        'img0': new_img0,
        'mask0': new_mask0,
        'img1': new_img1,
        'mask1': new_mask1,
        'camera': camera
    }
    return stereo_data

def load_data(cam, t):
    intr = np.array(calib_full[cam]['K']).astype(float).reshape((3, 3))
    dist = np.array(calib_full[cam]['distCoeff']).astype(float).reshape((5))
    img_sz = calib_full[cam]['imgSize']
    t_cam_name = '%s_%s.jpg' % (t, cam)
    file_name = os.path.join(ori_dir, t_cam_name)
    img = cv2.imread(file_name)
    img = cv2.undistort(img, intr, dist, None)
    R_ = np.array(calib_full[cam]['R']).astype(float).reshape((3, 3))
    T_ = np.array(calib_full[cam]['T']).astype(float).reshape((3, 1))
    extr = np.concatenate([R_, T_], 1)
    return (img, intr, extr), img_sz 

def save_np_to_json(parm, save_name, img_sz):
    w,h = img_sz
    move_t = (h-w)/2
    parm['intr0'][1,-1] -= move_t
    parm['intr1'][1,-1] -= move_t
    rescale = 1024/min(w, h)
    parm['intr0'][:2] *= rescale
    parm['intr1'][:2] *= rescale
    parm['Tf_x'] *= rescale 
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)
        

if __name__ == '__main__':
    # python step_0rect.py -i s2a3 -t val
    data_root = '/home/hjx/data4T/raw_thumanmv/' # TODO
    out_dir = '/home/hjx/data4T/raw_thumanmv/' # TODO

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input sequence')
    parser.add_argument('-t', '--trainval', type=str, required=True, help='train or val')
    arg = parser.parse_args()

    data_n = arg.input 
    ori_dir = data_root+data_n
    out_dir += arg.trainval
    calib_path = ori_dir+'/calibration_full.json'
    
    file_list = sorted(os.listdir(ori_dir))
    used_time_id_list = []

    for file in tqdm(file_list):
        if file[-3:] != 'jpg':
            continue
        
        time_id = file.split('/')[-1].split('_')[0]
        if time_id not in used_time_id_list:
            drop_flag = False

            if not drop_flag:
                used_time_id_list.append(time_id)
                
    if arg.trainval == 'train':
        used_time_id_list = sorted(used_time_id_list)[:300]
    elif arg.trainval == 'val':
        used_time_id_list = sorted(used_time_id_list)[50:80]
    elif arg.trainval == 'test':
        used_time_id_list = sorted(used_time_id_list)
        out_dir = out_dir + '/' + data_n + '_process'
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        shutil.copyfile(calib_path, out_dir+'/calibration_full.json')
    else:
        exit()

    img_dir = os.path.join(out_dir, 'img')
    Path(img_dir).mkdir(exist_ok=True, parents=True)

    msk_dir = os.path.join(out_dir, 'mask')
    Path(msk_dir).mkdir(exist_ok=True, parents=True)

    par_dir = os.path.join(out_dir, 'parameter')
    Path(par_dir).mkdir(exist_ok=True, parents=True)
    
    cam_id_list_s = [
        [
        '22139908',
        '22139909'],
        [
        '22139909',
        '22139914'],
        [
        '22139914',
        '22139906']
        ]

    for cam_id_list in cam_id_list_s:
        if cam_id_list[0] == '22139908':
            scene_n = 's1'
        elif cam_id_list[0] == '22139909':
            scene_n = 's2'
        elif cam_id_list[0] == '22139914':
            scene_n = 's3'
        else:
            print('wrong')
            exit()

        
        with open(calib_path, 'r') as f:
            calib_full = json.load(f)
        
        for t in tqdm(used_time_id_list):
            t_dir = os.path.join(img_dir, '%s_%s_%04d'%(data_n, scene_n, int(t)))
            t_par_dir = os.path.join(par_dir, '%s_%s_%04d'%(data_n, scene_n, int(t)))
            t_msk_dir = os.path.join(msk_dir, '%s_%s_%04d'%(data_n, scene_n, int(t)))
            if not os.path.exists(t_dir):
                os.mkdir(t_dir)
            if not os.path.exists(t_par_dir):
                os.mkdir(t_par_dir)
            if not os.path.exists(t_msk_dir):
                os.mkdir(t_msk_dir)
                
            out_path = os.path.join(t_dir, '%d.jpg')
            out_msk_path = os.path.join(t_msk_dir, '%d.jpg')
            out_par_path = os.path.join(t_par_dir, '%d_%d.json')

            mview, img_sz = load_data(cam_id_list[0], t)
            rview, img_sz = load_data(cam_id_list[1], t)

            rect_data = get_rectified_stereo_data(mview, rview, img_sz)
            img0 = rect_data['img0']
            img1 = rect_data['img1']

            msk0 = rect_data['mask0']
            msk1 = rect_data['mask1']
            
            w, h = img_sz[0], img_sz[1]
            # 1500, 2048
            move_t = (h-w)//2 
            
            
            ######## scene specific ###########

            img0 = img0[(move_t):(w + move_t), :, :] 
            img1 = img1[(move_t):(w + move_t), :, :] 

            msk0 = msk0[(move_t):(w + move_t), :, :] 
            msk1 = msk1[(move_t):(w + move_t), :, :] 
            
            ######## scene specific ###########

            img0 = cv2.resize(img0, (1024, 1024))
            img1 = cv2.resize(img1, (1024, 1024))
            msk0 = cv2.resize(msk0, (1024, 1024))
            msk1 = cv2.resize(msk1, (1024, 1024))
            
            cv2.imwrite(out_path % 0, img0.astype(np.uint8))
            cv2.imwrite(out_path % 1, img1.astype(np.uint8))

            cv2.imwrite(out_msk_path % 0, msk0.astype(np.uint8))
            cv2.imwrite(out_msk_path % 1, msk1.astype(np.uint8))
            
            save_np_to_json(rect_data['camera'], out_par_path % (0,1), img_sz)