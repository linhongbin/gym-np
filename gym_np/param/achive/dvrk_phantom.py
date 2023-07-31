import numpy as np
from gym_np.param.default import default
from copy import deepcopy
dvrk_phantom_param = deepcopy(default)
dvrk_phantom_param.suture_env.update({
"robot_type":'dvrk_2_0',
"cam_width" : 600,
"cam_height" : 600,
"cam_width_crop": [0.0,1.0],
"cam_height_crop": [0.23,0.55],
# "cam_height_crop": [0.0,1],
"cam_depth_z_min" : 230,
"cam_depth_z_max" : 320,
"ws_x_low":-0.014,
"ws_x_high":0.025,
"ws_y_low":-0.043,
"ws_y_high":0.013,
"ws_z_low":-0.161396,
"ws_z_high":-0.132,
"action_dx_max":0.002,
"action_dy_max":0.002,
"action_dz_max":0.002,
"init_pos_bound_dict":{ 'x':[0.001, 0.020],'y':[-0.036, -0.000],'z':[-0.14519, -0.14519]},
"init_orgin_RPY":[0,0,0, np.pi,0,0],
"q_dsr_reset":[0.088,0.161,0.137,0.000,-0.161,-0.088,],
"manual_set_base_rpy":[0,0,0, 0,0,0.06578527937021103],
"q_margin_ratio": 0.1,
"gripper_pos_min":-np.pi/18,
"gripper_pos_max":np.pi/4,
})


dvrk_phantom_param.NeedlePicking_wrapper.update({
"needle_init_pose_bound":{ 'low':[0.011, -0.026,-0.150, 0,0, -np.pi,],
              'high':[0.010, -0.010,-0.150, 0,0., np.pi,],},
})

dvrk_phantom_param.SimpleFSM_wrapper.update({
"method":"sensor",
})

