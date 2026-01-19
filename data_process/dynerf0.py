import cv2
import os
from glob import glob 
import numpy as np
from tqdm import tqdm 
import json

data_path = 'flame_salmon_1/' #TODO

cams = glob(data_path+'*.mp4')
cams = sorted([c_i[-6:-4] for c_i in cams])
print(cams)


pose_bounds = np.load(data_path+'poses_bounds.npy')
# print(pose.shape)
pose = pose_bounds[:, :-2].reshape(-1, 3, 5)
R = pose[..., :3]
T = pose[..., 3:4]

H_W_F = pose[..., 4:].reshape(-1, 3)
# print(R, T, H_W_F)
cam_num = R.shape[0]
pose_np = np.zeros([cam_num, 4, 4])
w2c = np.zeros([cam_num, 4, 4])

pose_cv_np = np.zeros([cam_num, 4, 4])
w2c_cv = np.zeros([cam_num, 4, 4])
for i in range(cam_num):
    pose_np[i, :3, :3] = R[i]
    pose_np[i, :3, 3:4] = T[i]
    pose_np[i, 3, 3] = 1

    w2c[i] = np.linalg.inv(pose_np[i])

    pose_cv_np[i, :3, :3] = R[i]
    print(pose_cv_np.shape)
    pose_cv_np[i, :3, :3] = np.concatenate([pose_cv_np[i,:3, 1:2], pose_cv_np[i, :3, 0:1], -pose_cv_np[i, :3, 2:3]], -1)
    pose_cv_np[i, :3, 3:4] = T[i]
    pose_cv_np[i, 3, 3] = 1
    print(pose_cv_np.shape)
    w2c_cv[i] = np.linalg.inv(pose_cv_np[i])

R = w2c[...,:3, :3]
print(R.shape)
T = w2c[..., :3,  3:4]

R_cv = w2c_cv[...,:3, :3]
print(R.shape)
T_cv = w2c_cv[..., :3,  3:4]



K = np.zeros((cam_num, 3, 3))
for i in range(cam_num):
    K_i = np.eye(3)
    K_i[0, 0] = H_W_F[i][2]
    K_i[1, 1] = H_W_F[i][2]

    K_i[0, 2] = H_W_F[i][1]/2
    K_i[1, 2] = H_W_F[i][0]/2
    K[i] = K_i


calib = {}
for i, cam_i in enumerate(cams):
    calib[cam_i] = {}
    calib[cam_i]['K'] = K[i].tolist()
    calib[cam_i]['R'] = R[i].tolist()
    calib[cam_i]['T'] = T[i].tolist()
    calib[cam_i]['imgSize'] = [H_W_F[i][1], H_W_F[i][0]]
with open(data_path+'./calib.json','w') as f:
    json.dump(calib,f, indent=4)


calib_cv = {}
for i, cam_i in enumerate(cams):
    calib_cv[cam_i] = {}
    calib_cv[cam_i]['K'] = K[i].tolist()
    # poses = np.concatenate([R[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :]], 1)
    calib_cv[cam_i]['R'] = R_cv[i].tolist()
    calib_cv[cam_i]['T'] = T_cv[i].tolist()
    calib_cv[cam_i]['imgSize'] = [H_W_F[i][1], H_W_F[i][0]]
with open(data_path+'./calibration_full.json','w') as f:
    json.dump(calib_cv,f, indent=4)

for cam_n, cam_s in enumerate(cams):
  track_n = 'cam%s.mp4' % cam_s 
  vidcap = cv2.VideoCapture(data_path + track_n)
  out_path = data_path + track_n[:-4]
  if not os.path.exists(out_path):
    os.mkdir(out_path)
  
  
  ## video process 

  success,image = vidcap.read()
  count = 0
  success = True
  while success:
    success,image = vidcap.read()
    if type(image) is np.ndarray:
      cv2.imwrite(out_path+'/frame%03d.jpg' % count, image)     # save frame as JPEG file
    else:
      break
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
    count += 1
