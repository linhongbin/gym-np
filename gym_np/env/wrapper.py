from gym_np.env.suture_env import SurgialChallengeEnv
from gym_np.tool.common import RPY2T, T2RPY, filter, filter_compare, fill_segment, scale, resize_img, PACKAGE_ROOT_PATH
from gym_np.tool.state_estimate import StateEstimator, TaskStates


from typing import List
import numpy as np
from PyKDL import Rotation
from numpy import pi
from time import sleep
import cv2
import gym
import time
from pathlib import Path
import ruamel.yaml as yaml

# from gym_np.calibrate import set_error
# from rospy import spin

radian_list = lambda x: np.deg2rad(np.array(x)).tolist()


class WrapperModified():
    def __init__(self, env, verbose=0):
        self.env = env
        self.verbose = verbose

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        return getattr(self.env, name)

    @property
    def seed(self):
        return self.env.seed

    @seed.setter
    def seed(self, seed):
        self.env.seed = seed

    def get_env_by_classname(self, class_name):
        if class_name == self.__class__.__name__:
            return self
        else:
            return self.env.get_env_by_classname(class_name)


class ClutchEngage(WrapperModified):
    """ Virtual Clutch """

    def __init__(self, env, start_engage_step=6, verbose=0):
        super().__init__(env, verbose)
        self.start_engage_step = start_engage_step

    def step(self, action, **kwargs):
        self.unwrapped.is_clutch_engage = self.env.timestep >= self.start_engage_step
        obs, reward, done, info = self.env.step(action, **kwargs)
        return obs, reward, done, info


class ImageResizer(WrapperModified):
    """ Resize image"""

    def __init__(self, env, resize_image_width=64, verbose=0):
        super().__init__(env, verbose)
        self.resize_image_width = resize_image_width

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs['image'] = resize_img(obs['image'], self.resize_image_width)
        return obs

    # def render(self, **kwargs):
    #     if kwargs["mode"] == "rgb_array":
    #         frame = self.env.render(**kwargs)
    #         frame = resize_img(frame, self.resize_image_width)
    #         return frame
    #     else:
    #         raise NotImplementedError

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs['image'] = resize_img(obs['image'], self.resize_image_width)
        return obs, reward, done, info


class Scaler2ImageEncoding(WrapperModified):
    """ encode scalar signals to image"""

    def __init__(self, env,
                 encoding_obs_key=["gripper_state",
                                   "state",
                                   "needle_box_x_rel",
                                   "needle_box_y_rel",
                                   "gripper_box_x_rel",
                                   "gripper_box_y_rel"],
                 fill_channel_idx=0,
                 verbose=0):
        super().__init__(env, verbose)
        self.encoding_obs_key = encoding_obs_key
        self.fill_channel_idx = fill_channel_idx

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.process(obs)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.process(obs)
        return obs, reward, done, info

    def process(self, obs):
        _high = []
        _low = []
        _value = []
        for k in self.encoding_obs_key:
            _obs_space = self.env.observation_space
            _high.append(_obs_space[k].high)
            _low.append(_obs_space[k].low)
            _value.append(obs[k])

        _high = np.concatenate(_high, axis=0)
        _low = np.concatenate(_low, axis=0)
        _value = np.concatenate(_value, axis=0)
        _value_norm = scale(_value, old_min=_low,
                            old_max=_high, new_min=0.0, new_max=255.0)
        _value_norm = _value_norm.astype(np.uint8)
        _value_norm = np.tile(_value_norm, (obs['image'].shape[0], 1))
        _value_norm = np.transpose(_value_norm)
        obs['image'][0:_value_norm.shape[0], :,
                     self.fill_channel_idx] = _value_norm
        return obs


class GripperObs(WrapperModified):
    """ encode gripper state"""

    def __init__(self, env, image_info=False, obs_keys=["gripper_state"], verbose=0):
        super().__init__(env, verbose)
        self.image_info = image_info
        self.obs_keys = obs_keys

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs['image'] = self.add_image_gripper_info(obs['image'])
        if "gripper_state" in self.obs_keys:
            obs['gripper_state'] = np.array(
                [self.env.is_gripper_close], dtype=np.float64)
        return obs

    # def render(self, **kwargs):
    #     if kwargs["mode"] == "rgb_array":
    #         frame = self.env.render(**kwargs)
    #         frame = self.add_image_gripper_info(frame)
    #         return frame
    #     else:
    #         raise NotImplementedError

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs['image'] = self.add_image_gripper_info(obs['image'])
        if "gripper_state" in self.obs_keys:
            obs['gripper_state'] = np.array(
                [self.env.is_gripper_close], dtype=np.float64)
        return obs, reward, done, info

    def add_image_gripper_info(self, im):
        if self.image_info:
            is_gripper_close = self.env.is_gripper_close
            im[0, 0, 2] = 255 if is_gripper_close else 0
        return im

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        if "gripper_state" in self.obs_keys:
            obs['gripper_state'] = gym.spaces.Box(low=np.float32(
                0*np.ones(1)), high=np.float32(1*np.ones(1)), dtype=np.float32)
        return gym.spaces.Dict(obs)


class PoseTracking(WrapperModified):
    """" track poses """

    def __init__(self, env,
                 is_drop_image=False,
                 is_scale=True,
                 neelde_pos_lim=[[-0.4, 0.36, 0.65], [0.0, 0.96, 1]],
                 gripper_pos_lim=[[-0.4, 0.36, 0.65], [0.0, 0.96, 1]],
                 verbose=0
                 ):
        super().__init__(env, verbose)
        self.is_drop_image = is_drop_image
        assert self.robot_type == 'ambf'
        self.neelde_pos_lim = neelde_pos_lim
        self.gripper_pos_lim = gripper_pos_lim
        self.neelde_pos_lim[0] = np.array(self.neelde_pos_lim[0] + [-np.pi]*3)
        self.neelde_pos_lim[1] = np.array(self.neelde_pos_lim[1] + [np.pi]*3)
        self.gripper_pos_lim[0] = np.array(
            self.gripper_pos_lim[0] + [-np.pi]*3)
        self.gripper_pos_lim[1] = np.array(self.gripper_pos_lim[1] + [np.pi]*3)

        self.is_scale = is_scale

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs.pop("depth_xyz", None)
        if self.is_drop_image:
            obs.pop('image', None)
        needle_pose, gripper_pose = self._track()
        obs['needle_pose'] = needle_pose
        obs['gripper_pose'] = gripper_pose
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs.pop("depth_xyz", None)
        if self.is_drop_image:
            obs.pop('image', None)
        needle_pose, gripper_pose = self._track()
        obs['needle_pose'] = needle_pose
        obs['gripper_pose'] = gripper_pose
        return obs

    def _track(self):
        info = self._get_info()  # this only works when using ambf
        needle_pose = np.concatenate(
            [info['needle_pos'], info['needle_rpy']], axis=0)
        gripper_pose = np.concatenate(
            [info['gripper_pos'], info['gripper_rpy']], axis=0)

        if self.is_scale:
            needle_pose = scale(
                needle_pose, self.neelde_pos_lim[0], self.neelde_pos_lim[1], -np.ones(6), np.ones(6))
            gripper_pose = scale(
                gripper_pose, self.gripper_pos_lim[0], self.gripper_pos_lim[1], -np.ones(6), np.ones(6))
        return needle_pose, gripper_pose

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        if self.is_drop_image:
            obs.pop('image', None)
        obs.update({'needle_pose': gym.spaces.Box(
            low=np.float32(-1*np.ones(6)), high=np.float32(1*np.ones(6)), dtype=np.float32)})
        obs.update({'gripper_pose': gym.spaces.Box(
            low=np.float32(-1*np.ones(6)), high=np.float32(1*np.ones(6)), dtype=np.float32)})
        return gym.spaces.Dict(obs)


class Simple_FSM(WrapperModified):
    """ Simple implementation of Finite State Machine """

    def __init__(self, env,
                 reward_inprogress_weight=-0.001,
                 reward_fail_ws_weight=-0.001,
                 reward_fail_jnt_weight=-0.001,
                 reward_fail_seg_weight=-0.001,
                 reward_fail_timelimit_weight=-0.001,
                 reward_sucess_weight=1,
                 method="seg_box",
                 method_args={"sucess_box_dist_thres": 0.007,
                              "sucess_sig_count": 2},
                 is_segfaildone=True,
                 is_wsfaildone=False,
                 is_qlimfaildone=False,
                 verbose=0
                 ):
        super().__init__(env, verbose)
        self.task_type = env.task_type
        self.robot_type = env.robot_type
        self.state_estimator = StateEstimator(task_type=self.task_type,
                                              robot_type=self.robot_type,
                                              method=method,
                                              method_args=method_args,
                                              is_force_inprogress=not is_segfaildone,
                                              verbose=self.verbose)
        self.reward_inprogress_weight = reward_inprogress_weight
        self.reward_sucess_weight = reward_sucess_weight
        self.is_segfaildone = is_segfaildone
        self.is_wsfaildone = is_wsfaildone
        self.is_qlimfaildone = is_qlimfaildone
        self.reward_fail_ws_weight = reward_fail_ws_weight
        self.reward_fail_jnt_weight = reward_fail_jnt_weight
        self.reward_fail_seg_weight = reward_fail_seg_weight
        self.reward_fail_timelimit_weight = reward_fail_timelimit_weight

    def step(self, action):
        obs, _, _, info = self.env.step(action)
        state = self.state_estimator(obs, info, action)
        obs['state'] = np.array([state.value])
        if state == TaskStates.InProcess:
            reward = self.reward_inprogress_weight
            done = False
        elif state in [
            TaskStates.EndFail_WorkSpace,
            TaskStates.EndFail_JointLimit,
            TaskStates.EndFail_Segment,
            TaskStates.EndFail_TimeLimit,
        ]:
            reward = {TaskStates.EndFail_WorkSpace: self.reward_fail_ws_weight,
                      TaskStates.EndFail_JointLimit: self.reward_fail_jnt_weight,
                      TaskStates.EndFail_Segment: self.reward_fail_seg_weight,
                      TaskStates.EndFail_TimeLimit: self.reward_fail_timelimit_weight}[state]
            done = True
            # print("=====")
            # # print(f"done: {info['done']},")
            # print(f"detect state {state}")
            # print("=====")
            if state == TaskStates.EndFail_Segment and not self.is_segfaildone:
                done = False
            if state == TaskStates.EndFail_WorkSpace and not self.is_wsfaildone:
                done = False
            if state == TaskStates.EndFail_JointLimit and not self.is_qlimfaildone:
                done = False

        elif state == TaskStates.EndSucess:
            reward = self.reward_sucess_weight
            done = True
            info['done'].append({"success": None})
            _env = self.get_env_by_classname("NeedlePicking")
            _env.is_needle_grasped = True
        else:
            raise Exception
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.state_estimator.reset()
        obs['state'] = np.array([TaskStates.InProcess.value])
        return obs

    @property
    def observation_space(self):
        """ 
        refer to
        https://github.com/jsikyoon/dreamer-torch/blob/main/wrappers.py
        https://github.com/ryanhoque/fabric-vsf/blob/master/gym_cloth/envs/cloth_env.py
        """
        obs = {k: v for k, v in self.env.observation_space.items()}
        obs['state'] = gym.spaces.Box(low=np.float32(0*np.ones(1)),
                                      high=np.float32((len(TaskStates)-1)*np.ones(1)), dtype=np.float32)
        return gym.spaces.Dict(obs)


class TimeLimit(WrapperModified):
    """ timelimit in a rollout """

    def __init__(self, env,
                 max_timestep=100,
                 verbose=0
                 ):
        super().__init__(env, verbose)
        self.max_timestep = max_timestep

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        is_exceed = self.timestep >= self.max_timestep
        if is_exceed:
            _info = {'timelimit_out': None}
            info['done'].append(_info)
        return obs, reward, done, info


class Visualizer(WrapperModified):
    """ render image with GUI """

    def __init__(self, env, is_key_blocking=True, verbose=0, save_dir="./data/vis/", vis_channel_idx=[0, 1, 2], is_gray_style=False) -> None:
        super().__init__(env, verbose)
        self.is_active = True
        self.is_key_blocking = is_key_blocking
        self.obs_image = None
        self.save_dir = Path(save_dir)
        self.vis_channel_idx = vis_channel_idx
        self.is_gray_style = is_gray_style

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        self.obs_image = obs["image"]
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.obs_image = obs["image"]
        return obs

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.obs_image

        elif mode in ['human', 'human_depth']:
            while True:
                frame = self.obs_image
                # print(frame.shape)
                # print(frame[0:10,0,0])
                frame1 = np.zeros(frame.shape, dtype=np.uint8)
                frame1[:, :, self.vis_channel_idx] = frame[:,
                                                           :, self.vis_channel_idx]
                frame = frame1
                if self.is_gray_style:
                    frame1 = np.zeros(frame.shape, dtype=np.uint8)
                    for i in range(frame.shape[2]):
                        frame1 = np.clip(
                            frame1 + np.stack([frame[:, :, i]]*3, axis=2), 0, 255)
                    frame = frame1
                frame = cv2.resize(frame, (600, 600),
                                   interpolation=cv2.INTER_NEAREST)
                # Display the resulting frame
                cv2.imshow('preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.setWindowTitle(
                    'preview', 'press q to exit, press other keys to step')

                # Display the resulting frame
                cv2.imshow('preview', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.setWindowTitle(
                    'preview', 'press q to exit, press other keys to step')

                _t = 0 if self.is_key_blocking else 500
                k = cv2.waitKey(_t)
                if k & 0xFF == ord('q'):    # Esc key to stop
                    print("press q ..")
                    self.close()
                    break
                elif k & 0xFF == ord('s'):    # s key to save pic
                    print("press s key, save pic...")
                    self.save_dir.mkdir(exist_ok=True, parents=True)
                    file = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(str(self.save_dir / file) + '.png',
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                elif k & 0xFF == ord('g'):    # g key to gray
                    self.is_gray_style = not self.is_gray_style
                    print(f"gray style: {self.is_gray_style}")
                else:
                    # print("")
                    break

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()
        self.is_active = False


def action_fixRY2all(action):
    _action = np.zeros(7)
    _action[:3] = action[:3]
    _action[5:] = action[3:]
    return _action


def action_all2fixRY(action):
    _action = np.zeros(5)
    _action[:3] = action[:3]
    _action[3:] = action[5:]
    return _action


class ImagePreprocess(WrapperModified):
    """ preprocess raw images """

    def __init__(self, env, robot_type,
                 preprocess_type="segment_script",
                 image_type="depth_seg_zoom_scalar",
                 info_keys=["needle_x_mean",
                            "needle_y_mean",
                            "needle_value_mean",
                            "needle_area",
                            "gripper_x_mean",
                            "gripper_y_mean",
                            "gripper_value_mean",
                            "gripper_area",
                            ],
                 obs_keys=["image",
                           "gripper_box_center_pos",
                           "needle_box_center_pos",
                           ],
                 segment_net_file=None,
                 is_save_anomaly_pic=True,
                 verbose=0,
                 depth_gaussion_noise_scale=0.2,
                 zoom_margin_ratio=0.3,
                 ):
        super().__init__(env, verbose)
        assert preprocess_type in ['segment_script', 'segment_net']
        self.is_depth = self.unwrapped.is_depth
        self.preprocess_type = preprocess_type
        if preprocess_type in ['segment_net', 'segment_script']:
            from gym_np.tool.segment import SegmentEngine
            self.segment_engine = SegmentEngine(process_type=self.preprocess_type,
                                                robot_type=robot_type,
                                                image_type=image_type,
                                                segment_net_file=segment_net_file,
                                                zoom_margin_ratio=zoom_margin_ratio)
        self.info_keys = info_keys
        self.obs_keys = obs_keys
        self.depth_gaussion_noise_scale = depth_gaussion_noise_scale

    def reset(self, **kwargs):
        _env = self.get_env_by_classname("NeedlePicking")
        obs = self.env.reset(**kwargs)
        depth_xyz = obs['depth_xyz'] if self.is_depth else None
        results = self.process(obs['image'], depth_xyz=depth_xyz)
        if _env.reset_needle_mode == "auto":
            while True:
                if "needle_box_x" in results:
                    break
                _env.reset_needle_mode = "manual"
                obs = self.env.reset(**kwargs)
                depth_xyz = obs['depth_xyz'] if self.is_depth else None
                results = self.process(obs['image'], depth_xyz=depth_xyz)

        obs = {}  # clear obs object
        for k in self.obs_keys:
            obs[k] = results[k]
        obs.pop("depth_xyz", None)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        depth_xyz = obs['depth_xyz'] if self.is_depth else None
        results = self.process(obs['image'], depth_xyz=depth_xyz)
        info.update({k: v for k, v in results.items() if k in self.info_keys})
        if not results['is_success']:
            _info = {'segment_fail': None
                     }
            info['done'].append(_info)
        obs = {}  # clear obs object
        for k in self.obs_keys:
            obs[k] = np.array(list(results[k]))
        return obs, reward, done, info

    def process(self, im, depth_xyz=None):
        results = {}
        depth = depth_xyz
        if self.depth_gaussion_noise_scale > 0:
            noise = self.depth_gaussion_noise_scale * \
                np.random.normal(0, 1, depth.shape) * 255
            depth = np.clip(depth + noise, 0, 255)
            depth = depth.astype(np.uint8)
        # if self.preprocess_type == 'mixdepth':
        #     im_pre = im
        #     if self.is_depth:
        #         results['image'] = np.concatenate([depth,im_pre[:,:,1:]], axis=2)
        #         results['is_success'] = True
        #         return results
        results.update(self.segment_engine.process_image(im, depth=depth))
        return results

    @property
    def observation_space(self):
        obs = {k: v for k, v in self.env.observation_space.items()}
        # print(obs)
        result = self.segment_engine.default_result_stat()
        for k in self.obs_keys:
            if k != "image":
                _size = len(result[k])
                obs[k] = gym.spaces.Box(low=np.float32(
                    0*np.ones(_size)), high=np.float32(1*np.ones(_size)), dtype=np.float32)
        return gym.spaces.Dict(obs)

    @property
    def seed(self):
        return self.env.seed

    @seed.setter
    def seed(self, seed):
        self.env.seed = seed
        self.rng_image_noise = np.random.RandomState(
            seed+11)  # avoid using the same with child env


class DualShock_Oracle_Discrete(WrapperModified):
    """ input demonstrations with dualshock controllers """

    def __init__(self, env,
                 sig_keys=None,
                 verbose=0):
        super().__init__(env, verbose)
        from gym_np.tool.input import DS_Controller
        self._con = DS_Controller(sig_keys=sig_keys)
        try:
            _ = env.action_space.n
        except:
            raise Exception('you need to use discrete action wrapper')

    def get_oracle_action(self, cmd_num=None):
        _cmd_num = cmd_num
        if cmd_num == None:
            _cmd_num = self.env.action_space.n
        self._con.led_on(b=1)
        value = self._con.get_discrete_cmd(_cmd_num+1)
        self._con.led_off()
        return value

    def reset(self, **kwargs):
        _env = self.get_env_by_classname("NeedlePicking")
        if _env.reset_needle_mode == "manual" and self.unwrapped.robot_type == 'dvrk_2_0':
            self.unwrapped.open_gripper()
            print("please reset needle, and then press ds button")
            self._con.led_on(r=1)
            _ = self._con.get_discrete_cmd(self.env.action_space.n)
            self._con.led_off()
        return self.env.reset(**kwargs)

    def get_ds_cmd(self):
        return self._con.get_discrete_cmd()


class ActionMask(WrapperModified):
    """ mask actions """

    def __init__(self, env, mask_index_list: List, verbose=0):
        super().__init__(env, verbose)

        self.mask_index_list = mask_index_list
        self.mask_arr = np.ones(self.env.action_space.shape, dtype=np.int32)
        self.mask_arr[mask_index_list] = 0

    def mask_forward(self, action):
        return action[self.mask_arr == 1]

    def mask_backward(self, action):
        _action = np.zeros(self.mask_arr.shape)
        _action[self.mask_arr == 1] = action
        return _action

    def step(self, action, **kwargs):
        return self.env.step(self.mask_backward(action), **kwargs)

    def get_oracle_action(self, **kwargs):
        return self.mask_forward(self.env.get_oracle_action(**kwargs))

    @property
    def action_space(self):
        n = self.env.action_space.shape[0] - len(self.mask_index_list)
        return gym.spaces.Box(low=np.float32(-1*np.ones(n)), high=np.float32(1*np.ones(n)), dtype=np.float32)


class DiscreteAction(WrapperModified):
    """ discretize actions """

    def __init__(self, env,
                 is_idle_action=False,
                 is_done_action=False,
                 verbose=0):
        super().__init__(env, verbose)
        self.action_dim = self.env.action_space.shape[0]
        assert self.action_dim == 5  # onlys support x,y,z,yaw,gripper
        self.action_list = []
        self.is_gripper_close = False
        self.is_idle_action = is_idle_action
        self.is_done_action = is_done_action
        self.cnt = 0
        self.action_strs = ['x_neg', 'x_pos', 'y_neg', 'y_pos', 'z_neg',
                            'z_pos', 'rot_neg', 'rot_pos', 'gripper_toggle', 'idle', 'done']
        self.action_prim = {}  # store discrete action primitives
        for i in range(4):
            _action = np.zeros(self.action_dim)
            _action[i] = -1
            self.action_prim[self.action_strs[2*i]] = _action
            _action = np.zeros(self.action_dim)
            _action[i] = 1
            self.action_prim[self.action_strs[2*i+1]] = _action
        self.action_prim['gripper_toggle'] = np.zeros(self.action_dim)
        self.action_prim['idle'] = np.zeros(self.action_dim)
        self.action_prim['done'] = np.zeros(self.action_dim)

        if self.is_idle_action and self.is_done_action:
            self.action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif self.is_idle_action and not self.is_done_action:
            self.action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif not self.is_idle_action and self.is_done_action:
            self.action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
        else:
            # print("XXXXXXXXxxxx========12sdfasdfasdfsdf")
            if self.unwrapped.robot_type == "ambf":
                self.action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            else:
                self.action_idx = [1, 0, 3, 2, 4, 5, 6, 7, 8]
        self.action_discrete_n = len(self.action_idx)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.action_discrete_n)

    def reset(self, **kwargs):
        obs = self.env.reset(gripper_pos=1, **kwargs)
        # _open =  np.zeros(self.action_dim)
        # _open[-1] = 1
        # _ = self.env.step(_open)
        self.is_gripper_close = False
        self.cnt = 0
        self.is_clutch_engage_prv = self.env.unwrapped.is_clutch_engage
        return obs

    def step(self, action, **kwargs):
        self.cnt += 1
        # print(f"step: {self.cnt} | act: [{action}] {self.action_strs[int(action)]}")

        _action = self.action_prim[self.action_strs[self.action_idx[action]]]
        if self.action_strs[self.action_idx[action]] == 'gripper_toggle':
            self.is_gripper_close = not self.is_gripper_close

        # starting step for clutch engaged
        if (not self.is_clutch_engage_prv) and (self.env.unwrapped.is_clutch_engage):
            self.is_gripper_close = False

        if self.is_gripper_close:
            _action[-1] = -1
        else:
            _action[-1] = 1

        obs, reward, done, info = self.env.step(_action, **kwargs)
        if self.action_strs[self.action_idx[action]] == 'done':
            done = True

        self.is_clutch_engage_prv = self.env.unwrapped.is_clutch_engage

        return obs, reward, done, info

    def get_oracle_action(self, **kwargs):
        action = self.env.get_oracle_action(**kwargs)
        ref = np.zeros(action.shape)
        if self.is_gripper_close:
            ref[-1] = -1
        else:
            ref[-1] = 1
        _err = action - ref
        # print("discrete err:", _err)
        if np.max(np.abs(_err)) < 0.5:
            if self.is_idle_action:
                _action = 9  # no motion
            else:
                _action = 0  # attach to other random action, try not to apear this case
        else:
            _index = np.argmax(np.abs(_err))
            _direction = _err[_index] > 0
            if _index == self.action_dim-1:  # gripper
                _action = 8
            else:
                if _direction:
                    _action = _index * 2 + 1
                else:
                    _action = _index * 2
        return _action


class NeedlePicking(WrapperModified):
    """ needle picking task """

    def __init__(self, env,
                 success_lift_height=0.008,
                 ecm_init_q=[0.0, 0, 0, 0],
                 # just mouse drag camera position, and hardcode position by this param
                 camL_local_pos=[0.14755337318467979, - \
                                 0.7046499973070069, -0.15954276945892427],
                 camL_local_quat=[-0.40890684301325836, - \
                                  0.40890663764030877, -0.5768842560352703, 0.5768839662958754],
                 waypoint_pos_acc_x=0.005,
                 waypoint_pos_acc_y=0.005,
                 waypoint_pos_acc_z=0.005,
                 waypoint_rot_acc=2,
                 needle_init_pose_bound={'low': [-0.27, 0.46, 0.75, 0, 0, -90],
                                         'high': [-0.2, 0.86,  0.75, 0, 0, 90]},
                 seed=0,
                 is_reset_cam=True,
                 is_idle_action=True,
                 reset_needle_mode="manual",
                 verbose=0
                 ):
        super().__init__(env, verbose)
        self.unwrapped.ecm_init_q = ecm_init_q
        self.camL_local_pos = camL_local_pos
        self.camL_local_quat = camL_local_quat
        self.waypoint_pos_acc_x = waypoint_pos_acc_x
        self.waypoint_pos_acc_y = waypoint_pos_acc_y
        self.waypoint_pos_acc_z = waypoint_pos_acc_z
        self.waypoint_rot_acc = np.deg2rad(waypoint_rot_acc)
        self.needle_init_pose_bound = {
            k: v[:3] + radian_list(v[3:]) for k, v in needle_init_pose_bound.items()}
        self.seed = seed
        self.is_reset_cam = is_reset_cam
        self.is_idle_action = is_idle_action
        self._init_var()

        self.success_lift_height = success_lift_height
        self.ERROR_MAG_ARR = np.deg2rad([5, 5, 0, 0, 0, 0])
        self.ERROR_MAG_ARR[2] = 0.05  # simulation unit, for insertion joint
        assert reset_needle_mode in ["auto", "manual"]
        self.reset_needle_mode = reset_needle_mode
        self.is_needle_grasped = False

    @property
    def task_type(self):
        return "needle_picking"

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.float32(-1*np.ones(7)), high=np.float32(1*np.ones(7)), dtype=np.float32)

    def _init_var(self):
        self.oracle_waypoint_list = None
        self.oracle_waypoint_cnt = 0
        self.direct_cnt = 0
        self.needle_initial_pos = None
        # self._is_grasp_prv = False
        self.step_cnt = 0

    def reset(self, **kwargs):

        action_arm_device = self.unwrapped.action_arm_device
        # =========move needle
        if self.reset_needle_mode == "manual":
            self.unwrapped.open_gripper()
            self.unwrapped.clients['world'].reset_bodies()
            needle_pose = self.rng_needle_init.uniform(
                self.needle_init_pose_bound['low'], self.needle_init_pose_bound['high']).tolist()
            self.clients['world'].set_needle_pose(
                needle_pose[:3], needle_pose[3:])
        elif self.reset_needle_mode == "auto":
            if self.is_needle_grasped:
                needle_pose = self.rng_needle_init.uniform(
                    self.needle_init_pose_bound['low'], self.needle_init_pose_bound['high']).tolist()
                _, rpy = T2RPY(self.unwrapped.init_orgin_T)
                needle_pose[3:5] = rpy[0:2]
                print(
                    f"needle pose {needle_pose} {self.needle_init_pose_bound['low']}")
                self.unwrapped.clients[action_arm_device].servo_tool_cp(
                    RPY2T(*needle_pose), 100)
                self.unwrapped.clients[action_arm_device].wait()
                sleep(0.5)
                self.unwrapped.open_gripper()
                self.unwrapped.clients[action_arm_device].wait()
        else:
            raise NotImplementedError

        # -=========== move cam
        if self.is_reset_cam:
            self.unwrapped.clients['world'].servo_cam_local(
                self.camL_local_pos, self.camL_local_quat)
        obs = self.env.reset(**kwargs)
        self._init_var()
        info = self._get_info()
        if 'needle_initial_pos' in info:
            self.needle_initial_pos = info['needle_initial_pos']
        return obs

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.env.seed = seed
        self.rng_needle_init = np.random.RandomState(
            seed+1)  # avoid using the same with child env

    def step(self, action):
        self.step_cnt += 1
        _action = action
        obs, reward, done, info = self.env.step(_action)
        info.update(self._get_info())
        return obs, reward, done, info

    def _get_info(self):
        info = {}
        device = self.unwrapped.action_arm_device
        info['is_grasp'] = int(
            self.unwrapped.clients[device].get_signal('measured_is_grasp'))
        if self.robot_type == 'ambf':
            pos, rpy = self.unwrapped.clients['world'].get_needle_pose()
            T_n_w = RPY2T(*pos, *rpy)
            T_g_w_msr = self.unwrapped.clients[device].T_g_w_msr
            info['needle_pos'], info['needle_rpy'] = T2RPY(T_n_w)
            info['gripper_pos'], info['gripper_rpy'] = T2RPY(T_g_w_msr)
            info['needle_initial_pos'] = info['needle_pos'] if self.needle_initial_pos is None else self.needle_initial_pos
            info['success_lift_height'] = self.success_lift_height

        # print(self.unwrapped.clients[device].get_signal('measured_is_grasp'))
        # print(obs['gripper_rpy'])
        # print("gripper_pos:", obs['gripper_pos'])
        # print("needle_pos:", obs['needle_pos'])
        return info

    def plan_oracle_waypoints(self):
        """ get plan waypoints """

        obs = self._get_info()
        T_gt_n = RPY2T(*[-0.09, 0.03, 0.00, 0, 0, 0])  # gt: gripper target
        T_pregrasp_hover = RPY2T(0, 0, 0.05, 0, 0, 0)
        T_hover = RPY2T(0, 0, 0.3, 0, 0, 0)

        _Y = obs['needle_rpy'][2]
        _offset_theta = pi/2
        _Y += _offset_theta

        # logic of gripper rotational direction
        if _Y > pi / 2:  # project to -pi to pi range
            _Y = _Y - pi
        elif _Y < -pi / 2:
            _Y = _Y + pi
        grasp_R = Rotation.RPY(*[0, 0, _Y]) * Rotation.RPY(*[pi, 0, 0])

        T_n_w = RPY2T(*obs['needle_pos'], *obs['needle_rpy'])
        T_gt_w = T_n_w * T_gt_n
        T_gt_w = RPY2T(*T_gt_w.p, *grasp_R.GetRPY())
        T_hover_w = T_hover * T_gt_w
        T_pregrasp_w = T_pregrasp_hover * T_gt_w

        waypoint_list = []
        waypoint_list.append(('goal', T2RPY(T_pregrasp_w) + (1,)))
        waypoint_list.append(('goal', T2RPY(T_gt_w) + (1,)))
        waypoint_list.append(('direct', [0, 0, 0, 0, 0, 0, -1]))
        waypoint_list.append(('goal', T2RPY(T_hover_w)+(-1,)))
        return waypoint_list

    def get_oracle_action(self, direct_repeat=4,
                          is_replan=False,
                          end_waypoint_action=[0, 0, 0, 0, 0, 0, -1],
                          noise_scale=0):
        """ get demonstration action"""

        _direct_repeat = direct_repeat if self.is_idle_action else 1

        if (self.oracle_waypoint_list is None) or is_replan:
            self.oracle_waypoint_list = self.plan_oracle_waypoints()

        for cnt in range(self.oracle_waypoint_cnt, len(self.oracle_waypoint_list), 1):
            # print("oracle wp No:", cnt)
            if self.oracle_waypoint_list[cnt][0] == 'direct':
                if self.direct_cnt >= _direct_repeat:
                    self.direct_cnt = 0
                    continue
                self.oracle_waypoint_cnt = cnt
                self.direct_cnt += 1
                action = np.array(
                    self.oracle_waypoint_list[cnt][1]) + self.rng.normal(0, noise_scale)

                output_action = np.clip(action,
                                        -np.ones(action.shape),
                                        np.ones(action.shape))
                return output_action

            else:
                obs = self._get_info()
                oracle_waypoint = self.oracle_waypoint_list[cnt][1]
                pos, rpy, grasp = oracle_waypoint[0], oracle_waypoint[1], oracle_waypoint[2]

                # correct joint error
                device = self.unwrapped.action_arm_device
                q_msr = np.array(
                    self.unwrapped.clients[device].get_signal('measured_js'))
                obs['gripper_pos'], obs['gripper_rpy'] = T2RPY(
                    self.unwrapped.clients[device].fk_map(q_msr.tolist()))

                # print(obs['gripper_pos'])
                pos_err = pos - obs['gripper_pos']
                rot_err = Rotation.RPY(*rpy.tolist()) * \
                    Rotation.RPY(*obs['gripper_rpy'].tolist()).Inverse()
                rpy_err = np.array(rot_err.GetRPY())

                # pos_err_clip = np.clip(np.linalg.norm(pos_err)-self.waypoint_pos_acc,a_min=0,a_max=None)
                # rot_err_clip = np.clip(rot_err.GetRotAngle()[0]-self.waypoint_rot_acc, a_min=0,a_max=None)
                # print(f'pos_err: {pos_err_clip} rot_err:{rot_err_clip}')

                # print("pos err",np.linalg.norm(pos_err))
                # print("rot err",np.rad2deg(rot_err.GetRotAngle()[0]))
                # print(obs['gripper_rpy'])
                # print(rpy)
                if np.abs(pos_err[0]) < self.waypoint_pos_acc_x and\
                        np.abs(pos_err[1]) < self.waypoint_pos_acc_y and\
                        np.abs(pos_err[2]) < self.waypoint_pos_acc_z and\
                        (rot_err.GetRotAngle()[0] < self.waypoint_rot_acc):
                    # print(f'Waypoint #{cnt} arrive')
                    continue  # within the accuracy of tollerance

                pos_max = [self.unwrapped.action_dx_max,
                           self.unwrapped.action_dy_max,
                           self.unwrapped.action_dz_max]
                rpy_max = [self.unwrapped.action_dR_max,
                           self.unwrapped.action_dP_max,
                           self.unwrapped.action_dY_max]
                pos_upper_arr = np.array(pos_max)
                rpy_upper_arr = np.array(rpy_max)

                rel_pos_err = np.divide(pos_err, pos_upper_arr)
                rel_rpy_err = np.divide(rpy_err, rpy_upper_arr)

                rel_pos_err_clip = np.clip(rel_pos_err,
                                           -np.ones(rel_pos_err.shape),
                                           np.ones(rel_pos_err.shape))
                rel_rpy_err_clip = np.clip(rel_rpy_err,
                                           -np.ones(rel_rpy_err.shape),
                                           np.ones(rel_rpy_err.shape))

                action = np.concatenate(
                    (rel_pos_err_clip, rel_rpy_err_clip, np.array([grasp])))

                self.oracle_waypoint_cnt = cnt
                noise = self.rng.normal(0, noise_scale)
                # print(f'noise:{noise} action:{action}')
                action = action + noise
                output_action = np.clip(action,
                                        -np.ones(action.shape),
                                        np.ones(action.shape))
                return output_action
        output_action = np.array(end_waypoint_action)
        return output_action

class GymSutureEnv(object):
    def __init__(self,robot_type="ambf",
                platform_type="cuboid",  # [cuboid, phantom]
                needle_type="standard",
                preprocess_type='segment_script',
                image_type="zoom_needle_gripper_boximage",
                scalar2image_obs_key=["gripper_state", "state"],
                action_arm_device='psm2',
                reset_needle_mode="auto",
                clutch_start_engaged=6,
                resize_resolution=-1,
                timelimit=-1,
                segment_net_file=None,
                is_depth=True,
                is_idle_action=False,
                is_done_action=False,
                is_ds4_oracle=False,
                is_visualizer=False,
                is_visualizer_blocking=True,
                is_dummy=False,
                is_segfaildone=True,
                is_save_anomaly_pic=True,
                verbose=0,
                depth_gaussion_noise_scale=0,
                ):
        # load default config
        default_file = Path(__file__).parent.parent / "param" / "default.yaml"
        config = yaml.safe_load(default_file.read_text())

        # load custom config and update
        custom_file = Path(__file__).parent.parent / "param" / (robot_type + ".yaml")
        custom_config = yaml.safe_load(custom_file.read_text())
        if custom_config is not None:
            for k, v in custom_config.items():
                config[k].update(v)

        # update needle config
        needle_name = "needle_" + needle_type
        for k, v in config[needle_name].items():
            config[k].update(v)

        env = SurgialChallengeEnv(
            is_depth=is_depth,
            action_arm_device=action_arm_device,
            verbose=verbose,
            is_dummy=is_dummy,
            **(config["suture_env"])
        )
        env = NeedlePicking(env,
                            verbose=verbose,
                            is_idle_action=is_idle_action,
                            reset_needle_mode=reset_needle_mode,
                            **(config["NeedlePicking_wrapper"])
                            )

        # common layers
        env = ActionMask(env, [3, 4])  # fix pitch yaw
        env = DiscreteAction(env, is_idle_action=is_idle_action,
                            is_done_action=is_done_action,
                            verbose=verbose,)

        if is_ds4_oracle:
            env = DualShock_Oracle_Discrete(env,)

        env = ImagePreprocess(env, robot_type=robot_type,
                            preprocess_type=preprocess_type,
                            image_type=image_type,
                            segment_net_file=segment_net_file,
                            is_save_anomaly_pic=is_save_anomaly_pic,
                            verbose=verbose,
                            depth_gaussion_noise_scale=depth_gaussion_noise_scale,
                            **(config["ImagePreprocess_wrapper"]),
                            )

        timelimit = int(timelimit)
        if timelimit > 0:
            env = TimeLimit(env, max_timestep=timelimit, verbose=verbose,)

        env = Simple_FSM(env, is_segfaildone=is_segfaildone,
                        verbose=verbose, **(config["SimpleFSM_wrapper"]))

        resize_resolution = int(resize_resolution)
        if resize_resolution > 0:
            env = ImageResizer(
                env, resize_image_width=resize_resolution, verbose=verbose,)

        env = GripperObs(env, verbose=verbose,)

        clutch_start_engaged = int(clutch_start_engaged)
        if clutch_start_engaged > 0:
            env = ClutchEngage(
                env, start_engage_step=clutch_start_engaged, verbose=verbose,)

        if len(scalar2image_obs_key) != 0:
            env = Scaler2ImageEncoding(
                env, encoding_obs_key=scalar2image_obs_key, verbose=verbose,)
        if is_visualizer:
            env = Visualizer(
                env, is_key_blocking=is_visualizer_blocking, verbose=verbose,)
        self._env = env

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        return getattr(self._env, name)


