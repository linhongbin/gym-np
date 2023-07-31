import numpy as np


class TuneParam():
    def __init__(self) -> None:
        self.suture_env = {}
        self.NeedlePicking_wrapper = {}
        self.SimpleFSM_wrapper = {}
        self.ImagePreprocess_wrapper = {}

default = TuneParam()
default.suture_env = {
"robot_type":'ambf',
"cam_width": 600,
"cam_height": 600,
"cam_width_crop": [0.0,0.92],
"cam_height_crop": [0.0,0.8],
"cam_depth_z_min" : -0.2,
"cam_depth_z_max" : 0.4,
"ws_x_low":-0.4, # compact workspace
"ws_x_high":0.0,
"ws_y_low":0.36,
"ws_y_high":0.96,
"ws_z_low":0.69,
"ws_z_high":1,
"action_dx_max":0.02,
"action_dy_max":0.02,
"action_dz_max":0.02,
"action_dR_max":10*2*np.pi/360,
"action_dP_max":10*2*np.pi/360,
"action_dY_max":10*2*np.pi/360,
"action_arm_itpl_num" : 5,
"action_jaw_itpl_num" : 5,
"init_pos_bound_dict":{'z':[0.852, 0.852]}, # starting needle z: 0.71172556  
"init_RPY_bound_dict":{'R':[0,0], # 10deg
                    'P':[0,0],
                    'Y':[np.deg2rad(-80), np.deg2rad(80)]},
"extra_delay_time":0.3,
"q_dsr_reset":[-0.5656515955924988, -0.15630173683166504, 1.3160043954849243, -2.2147457599639893, 0.8174221515655518,-1],
"q_margin_ratio": 0.1,
}

default.NeedlePicking_wrapper={
    "waypoint_pos_acc_x":0.015,
    "waypoint_pos_acc_y":0.015,
    "waypoint_pos_acc_z":0.015, 
    "waypoint_rot_acc":np.deg2rad(12), 
    "needle_init_pose_bound":{'low': [-0.3, 0.46, 0.75, 0,0, -np.pi], 
                            'high': [-0.2, 0.86,  0.75, 0,0, np.pi]},
    "camL_local_pos": [0.24755338507914557, -0.6809676085034938, -0.2572113011447541],
    "camL_local_quat": [-0.40890684301325825, -0.40890663764030877, -0.5768842560352703, 0.5768839662958752],
    "success_lift_height":0.06,
}

default.SimpleFSM_wrapper = {
                    "reward_inprogress_weight":-0.001,
                    "reward_fail_ws_weight":-0.01,
                    "reward_fail_jnt_weight":-0.001,
                    "reward_fail_seg_weight":-0.001,
                    "reward_fail_timelimit_weight":-0.1,
                    "reward_sucess_weight":1,
                    "method":"tracking",
                    # "method":"segment_stat",
                    # "method_args":{"sucess_box_dist_thres": 0.3,
                    #              "sucess_sig_count": 2,
                    #              "metric_weights": [40,40,30]},
                    }


default.ImagePreprocess_wrapper = {
    "zoom_margin_ratio": 0.3,
}