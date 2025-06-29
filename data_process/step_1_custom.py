
import os
import cv2
import glob
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import argparse 
from step_0rect_custom import load_cam_param
# python step_1_custom.py -t val

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

Path(processed_data_root).mkdir(exist_ok=True, parents=True)


file_list = sorted(os.listdir(ori_dir))
used_time_id_list = []

file_extension = None 
for file in tqdm(file_list):
    if file[-3:] != 'jpg' and file[-3:] != 'png' and file[-3:] != 'jpeg':
        continue
    
    time_id = file.split('/')[-1].split('_')[0]
    if time_id not in used_time_id_list:
        drop_flag = False

        if not drop_flag:
            used_time_id_list.append(time_id)
    if file_extension == None:
        file_extension = file[-3:]
        
'''
save new image
'''

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

par_dir = os.path.join(processed_data_root, 'parameter')
Path(par_dir).mkdir(exist_ok=True, parents=True)

n_set = (len(cam_names)-1)//(s_set-1) 
# make sure that you have the minimum number of cameras for at least one work set
cam_id_list_s = []
for set_i in range(n_set):
    cam_id_list_s.append(list(range(set_i*(s_set-1)+1, (set_i+1)*(s_set-1)))+[set_i*(s_set-1), (set_i+1)*(s_set-1)])

for set_i, cam_id_list in enumerate(cam_id_list_s):
    scene_n = 's%d'%(set_i+1)
        
    for cam_i, cam in enumerate(cam_id_list):
        w, h = img_size[0], img_size[1]
        
        tmp = intr.copy()
        extr = extrs[cam].copy()
        

        for t in tqdm(used_time_id_list):
            t_dir = os.path.join(img_dir, '%s_%04d'%(scene_n, int(t)))
            t_par_dir = os.path.join(par_dir, '%s_%04d'%(scene_n, int(t)))
            if not os.path.exists(t_dir):
                os.mkdir(t_dir)
            if not os.path.exists(t_par_dir):
                os.mkdir(t_par_dir)
            
            np.save(t_par_dir+'/%d_extrinsic.npy' % int(cam_i+2), extr)
            np.save(t_par_dir+'/%d_intrinsic.npy' % int(cam_i+2), tmp)

            t_cam_name = '%s_%s.%s' % (t, cam_names[cam], file_extension)
            file_name = os.path.join(ori_dir, t_cam_name)
            
            img = cv2.imread(file_name)

            out_path = os.path.join(t_dir, '%d.jpg' % int(cam_i+2))
            
            ######## scene specific ###########

            # img_tmp = dst[(move_t):(w + move_t), :, :] #3000, 3000

            ######## scene specific ###########

            # img_out = cv2.resize(img_tmp, (1024, 1024))
            cv2.imwrite(out_path, img.astype(np.uint8))
