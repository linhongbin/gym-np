from gym_suture.param.ambf_phantom import ambf_phantom_param
from gym_suture.param.dvrk_phantom import dvrk_phantom_param
import numpy as np
tune_params = {
    "ambf_phantom": ambf_phantom_param,
    "dvrk_phantom": dvrk_phantom_param,
}

def get_params(robot_platform, needle):
    _param = tune_params[robot_platform]
    if needle =="standard" or needle =="unregular1":
        pass
    elif needle =="large": # make sure needles are inside workspace
        _param.NeedlePicking_wrapper.update(
                {"needle_init_pose_bound":{'low': [-0.26, 0.5, 0.75, 0,0, -np.pi], 
                                            'high': [-0.24, 0.8,  0.75, 0,0, np.pi]},}
        )
    elif needle =="unregular2":
        _param.NeedlePicking_wrapper.update(
                {"needle_init_pose_bound":{'low': [-0.27, 0.49, 0.75, 0,0, -np.pi], 
                                            'high': [-0.23, 0.83,  0.75, 0,0, np.pi]},}
        )
    elif needle =="small":
        _param.NeedlePicking_wrapper.update(
                {
            "needle_init_pose_bound":{'low': [-0.32, 0.44, 0.75, 0,0, -np.pi], 
            'high': [-0.18, 0.88,  0.75, 0,0, np.pi]},
                }
        )    
    return _param

    # "needle_init_pose_bound":{'low': [-0.3, 0.46, 0.75, 0,0, -np.pi], 
    #                         'high': [-0.2, 0.86,  0.75, 0,0, np.pi]},