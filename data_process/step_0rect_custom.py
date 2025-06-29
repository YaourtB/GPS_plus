import numpy as np 
import os
import random
from PIL import Image
import cv2

from pathlib import Path
import json
import glob 
from tqdm import tqdm 
import argparse 
import shutil 
from colmap_read_model import read_cameras_binary, read_images_binary

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

def load_data(cam, t, f_extend = 'jpg'):
    t_cam_name = '%s_%s.%s' % (t, cam, f_extend)
    file_name = os.path.join(ori_dir, t_cam_name)
    img = cv2.imread(file_name)
    return img 

def load_cam_param(root_path):
    # for intrinisic
    camerasfile = os.path.join(root_path, 'sparse/0/cameras.bin')
    camdata = read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]

    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h, w, f]).reshape([3, 1])
    intr = np.eye(3)
    intr[0, 0] = f
    intr[1, 1] = f 
    intr[0, -1] = w/2 
    intr[1, -1] = h/2 
    img_size = w, h 
    
    # for extrinsic
    imagesfile = os.path.join(root_path, 'sparse/0/images.bin')
    imdata = read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('we assume that you have at least 4 (size of work set) cameras')
    print('actually, you have %d cameras'%len(names))
    perm = np.argsort(names) #??
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        # m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        
        extr = np.zeros((3,4))
        extr[:3, :3] = R
        extr[:3, 3:] = t
        w2c_mats.append(extr)
    return intr, w2c_mats, names, img_size 

def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)
        

if __name__ == '__main__':
    # python step_0rect_custom.py -t val
    data_root = '/PATH/TO/custom_data' # TODO
    processed_data_root = '/PATH/TO/processed_custom_data/' # TODO

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainval', type=str, required=True, help='train or val')
    parser.add_argument('-n', '--setsize', type=int, default=4, required=False, help='number of cameras for each work set')
    arg = parser.parse_args()

    ori_dir = data_root
    processed_data_root += arg.trainval
    s_set = arg.setsize
    
    intr, extrs, names, img_size = load_cam_param(ori_dir)
    cam_names = [n_i.split('.')[0].split('_')[1] for n_i in names]
    # we assume that you were using the images of views of the first frame to do colmap 
    # so n_i is in the format of "#frame_#cam.jpg" 
    # and we assume that all images are ordered from the leftmost to the rightmost or
    # from the rightmost to the leftmost

    file_list = sorted(os.listdir(ori_dir))
    used_time_id_list = []

    file_extension = None 
    for file in tqdm(file_list):
        if file[-3:] != 'png' and file[-3:] != 'jpg' and file[-3:] != 'jpeg':
            continue
        
        time_id = file.split('/')[-1].split('_')[0]
        if time_id not in used_time_id_list:
            drop_flag = False

            if not drop_flag:
                used_time_id_list.append(time_id)
        if file_extension == None:
            file_extension = file[-3:]
    
    total_length = len(used_time_id_list)
    if arg.trainval == 'train':
        # using the first 7/8 frames
        used_time_id_list = sorted(used_time_id_list)[:(-total_length//8)]
    elif arg.trainval == 'val':
        # using the last 1/8 frames
        used_time_id_list = sorted(used_time_id_list)[(-total_length//8):]
    else:
        exit()

    img_dir = os.path.join(processed_data_root, 'img')
    Path(img_dir).mkdir(exist_ok=True, parents=True)

    msk_dir = os.path.join(processed_data_root, 'mask')
    Path(msk_dir).mkdir(exist_ok=True, parents=True)

    par_dir = os.path.join(processed_data_root, 'parameter')
    Path(par_dir).mkdir(exist_ok=True, parents=True)
    
    n_set = (len(cam_names)-1)//(s_set-1) 
    # make sure that you have the minimum number of cameras for at least one work set
    cam_id_list_s = []
    for set_i in range(n_set):
        cam_id_list_s.append([set_i*(s_set-1), (set_i+1)*(s_set-1)])

    for set_i, cam_id_list in enumerate(cam_id_list_s):
        scene_n = 's%d'%(set_i+1)
        
        for t in tqdm(used_time_id_list):
            t_dir = os.path.join(img_dir, '%s_%04d'%(scene_n, int(t)))
            t_par_dir = os.path.join(par_dir, '%s_%04d'%(scene_n, int(t)))
            t_msk_dir = os.path.join(msk_dir, '%s_%04d'%(scene_n, int(t)))
            if not os.path.exists(t_dir):
                os.mkdir(t_dir)
            if not os.path.exists(t_par_dir):
                os.mkdir(t_par_dir)
            if not os.path.exists(t_msk_dir):
                os.mkdir(t_msk_dir)
                
            out_path = os.path.join(t_dir, '%d.jpg')
            out_msk_path = os.path.join(t_msk_dir, '%d.jpg')
            out_par_path = os.path.join(t_par_dir, '%d_%d.json')

            mimg = load_data(cam_names[cam_id_list[0]], t, file_extension)
            rimg = load_data(cam_names[cam_id_list[1]], t, file_extension)

            mview = mimg, intr.copy(), extrs[cam_id_list[0]].copy()
            rview = rimg, intr.copy(), extrs[cam_id_list[1]].copy()
            rect_data = get_rectified_stereo_data(mview, rview, img_size)
            img0 = rect_data['img0']
            img1 = rect_data['img1']

            msk0 = rect_data['mask0']
            msk1 = rect_data['mask1']
            
            
            ######## scene specific ###########

            # img0 = img0[(move_t):(w + move_t), :, :] 
            # img1 = img1[(move_t):(w + move_t), :, :] 

            # msk0 = msk0[(move_t):(w + move_t), :, :] 
            # msk1 = msk1[(move_t):(w + move_t), :, :] 
            
            ######## scene specific ###########

            # img0 = cv2.resize(img0, (1024, 1024))
            # img1 = cv2.resize(img1, (1024, 1024))
            # msk0 = cv2.resize(msk0, (1024, 1024))
            # msk1 = cv2.resize(msk1, (1024, 1024))
            
            cv2.imwrite(out_path % 0, img0.astype(np.uint8))
            cv2.imwrite(out_path % 1, img1.astype(np.uint8))

            cv2.imwrite(out_msk_path % 0, msk0.astype(np.uint8))
            cv2.imwrite(out_msk_path % 1, msk1.astype(np.uint8))
            
            save_np_to_json(rect_data['camera'], out_par_path % (0,1))
        
        extr0 = extrs[cam_id_list[0]].copy()
        extr1 = extrs[cam_id_list[1]].copy()
        cam_position0 = np.matmul(extr0[:3, 3:].T, extr0[:3, :3].T)
        cam_position1 = np.matmul(extr1[:3, 3:].T, extr1[:3, :3].T)
        dist_cam_set = np.linalg.norm(cam_position0-cam_position1)
        print('distance between the leftmost camera to the rightmost one in work set %d is '%(set_i+1), dist_cam_set)
        print('it is a hint to estimate inverse_depth_init in config/stage.yaml, try this ', 0.5/dist_cam_set)
