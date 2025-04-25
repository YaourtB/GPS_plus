
import os
import cv2
import glob
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse 

# python step_1.py -i s2a3 -t val

data_root = '/PATH/TO/raw_data/' # TODO
processed_data_root = '/PATH/TO/processed_data/' # TODO

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True, help='input sequence')
parser.add_argument('-t', '--trainval', type=str, required=True, help='train or val')
arg = parser.parse_args()

data_n = arg.input 
ori_dir = data_root+data_n
processed_data_root += arg.trainval


Path(processed_data_root).mkdir(exist_ok=True, parents=True)


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

'''
save new image
'''

if arg.trainval == 'train':
    used_time_id_list = sorted(used_time_id_list)[:300]
elif arg.trainval == 'val':
    used_time_id_list = sorted(used_time_id_list)[50:80]
elif arg.trainval == 'test':
    used_time_id_list = sorted(used_time_id_list)
    processed_data_root = processed_data_root + '/' + data_n + '_process'
else:
    exit()

img_dir = os.path.join(processed_data_root, 'img')
Path(img_dir).mkdir(exist_ok=True, parents=True)

par_dir = os.path.join(processed_data_root, 'parameter')
Path(par_dir).mkdir(exist_ok=True, parents=True)

cam_move = [0, 0, 0, 0] #TODO

cam_id_list_s = [
    [
    '22139907',
    '22070932',
    '22139908',
    '22139909'
    ],
    [
    '22053927',
    '22053908',
    '22139909',
    '22139914'
    ],
    [
    '22053925',
    '22053923',
    '22139914',
    '22139906'
    ]
]

for cam_id_list in cam_id_list_s:
    if cam_id_list[0] == '22139907':
        scene_n = 's1'
    elif cam_id_list[0] == '22053927':
        scene_n = 's2'
    elif cam_id_list[0] == '22053925':
        scene_n = 's3'
    else:
        exit()
        
    calib_path = ori_dir+'/calibration_full.json'
    with open(calib_path, 'r') as f:
        calib_full = json.load(f)
        
    for cam_i, cam in enumerate(cam_id_list):
        K = np.array(calib_full[cam]['K']).astype(float).reshape((3, 3))
        dist = np.array(calib_full[cam]['distCoeff']).astype(float).reshape((5))
        img_sz = calib_full[cam]['imgSize']
        in_mat = K
        w, h = img_sz[0], img_sz[1]
        # 1500, 2048
        move_t = (h-w)//2 + cam_move[cam_i]

        tmp = np.array(calib_full[cam]['K']).astype(float).reshape((3, 3)).copy()

        tmp[1,-1] -= move_t 
        
        ######## scene specific ###########

        
        tmp[:2] /= (min(w, h)/1024.0)
        calib_full[cam]['K'] = tmp.reshape(-1).tolist()
        calib_full[cam]['distCoeff'] = np.zeros(5).tolist()
        calib_full[cam]['imgSize'] = [1024, 1024]
        R_ = np.array(calib_full[cam]['R']).astype(float).reshape((3, 3))
        T_ = np.array(calib_full[cam]['T']).astype(float).reshape((3, 1))
        extr = np.concatenate([R_, T_], 1)
        print('intr cam ', cam)
        print(tmp)
        print('-----------------------')


        for t in tqdm(used_time_id_list):
            t_dir = os.path.join(img_dir, '%s_%s_%04d'%(data_n, scene_n, int(t)))
            t_par_dir = os.path.join(par_dir, '%s_%s_%04d'%(data_n, scene_n, int(t)))
            if not os.path.exists(t_dir):
                os.mkdir(t_dir)
            if not os.path.exists(t_par_dir):
                os.mkdir(t_par_dir)
            
            np.save(t_par_dir+'/%d_extrinsic.npy' % int(cam_i+2), extr)
            np.save(t_par_dir+'/%d_intrinsic.npy' % int(cam_i+2), tmp)

            t_cam_name = '%s_%s.jpg' % (t, cam)
            file_name = os.path.join(ori_dir, t_cam_name)
            img = cv2.imread(file_name)
            
            dst = cv2.undistort(img, in_mat, dist, None)
            out_path = os.path.join(t_dir, '%d.jpg' % int(cam_i+2))
            
            ######## scene specific ###########

            img_tmp = dst[(move_t):(w + move_t), :, :] #3000, 3000

            ######## scene specific ###########

            img_out = cv2.resize(img_tmp, (1024, 1024))
            cv2.imwrite(out_path, img_out.astype(np.uint8))
