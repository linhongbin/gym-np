"""
Gym environment for surgical robotic challenge
"""
from gym_np.tool.ros_client import PSMClient, ECMClient, WolrdClient
from gym_np.tool.common import resize_img, crop2square_img, RPY2T, PointCloud2_2_xyzNimage, scale_rawdepth_2_uint8depth, convert_xyz_2_depth_image, scale

import gym
import numpy as np
from gym import spaces
from numpy import pi
from PyKDL import Frame
import rospy
from rospy.timer import sleep
import cv2


class SurgialChallengeEnv(gym.Env):
    def __init__(self, robot_type='ambf',
                 cam_width=64,
                 cam_height=64,
                 cam_width_crop=[0.0, 1],
                 cam_height_crop=[0.0, 1.0],
                 cam_depth_z_min=200,
                 cam_depth_z_max=300,
                 obs_type='image',
                 obs_cam_type='rgb',
                 obs_cam_device='cameraL',
                 action_type='delta_motion',
                 action_arm_device='psm2',
                 action_dx_max=0.01,
                 action_dy_max=0.01,
                 action_dz_max=0.01,
                 action_dR_max=2,  # Deg
                 action_dP_max=2,
                 action_dY_max=2,
                 ws_x_low=-0.4,
                 ws_x_high=0.4,
                 ws_y_low=-0.9,
                 ws_y_high=0.9,
                 ws_z_low=-1.0,
                 ws_z_high=1.4,
                 init_pos_bound_dict={},
                 init_RPY_bound_dict={},
                 q_margin_ratio=0.1,
                 extra_delay_time=0.2,
                 cmd_send_elapse_time=0.01,
                 action_arm_itpl_num=4,
                 action_jaw_itpl_num=3,
                 #    is_done_out_ws = True,
                 #    is_done_out_jnt = False,
                 q_dsr_reset={'dvrk_2_0': [0, 0, 0.12, 0, 0, 0],
                                'ambf': [-0.5066242659894763, -0.11420078566782467, 1.3373470787079902, -2.136101385011827, 0.7932006961890031, -0.9020190481228663]},
                 seed=0,
                 init_orgin_RPY=[0, 0, 0, 180, 0, 0],
                 is_random_init_pose=True,
                 is_depth=True,
                 manual_set_base_rpy=[0, 0, 0, 0, 0, 0],
                 gripper_pos_min=0,
                 gripper_pos_max=45,  # Deg
                 depth_image_max=255,
                 depth_image_min=80,
                 is_dummy=False,
                 verbose=0,
                 ):
        radian_list = lambda x: np.deg2rad(np.array(x)).tolist()
        self.robot_type = robot_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_width_crop = cam_width_crop
        self.cam_height_crop = cam_height_crop
        self.obs_cam_type = obs_cam_type
        self.obs_cam_device = obs_cam_device
        self.action_type = action_type
        self.action_arm_device = action_arm_device
        self.action_dx_max = action_dx_max
        self.action_dy_max = action_dy_max
        self.action_dz_max = action_dz_max
        self.action_dR_max = np.deg2rad(action_dR_max)
        self.action_dP_max = np.deg2rad(action_dP_max)
        self.action_dY_max = np.deg2rad(action_dY_max)
        self.ws_x_low = ws_x_low
        self.ws_x_high = ws_x_high
        self.ws_y_low = ws_y_low
        self.ws_y_high = ws_y_high
        self.ws_z_low = ws_z_low
        self.ws_z_high = ws_z_high
        self.q_margin_ratio = q_margin_ratio
        self.extra_delay_time = extra_delay_time
        self.cmd_send_elapse_time = cmd_send_elapse_time
        self.action_arm_itpl_num = action_arm_itpl_num
        self.action_jaw_itpl_num = action_jaw_itpl_num
        self.q_dsr_reset = q_dsr_reset[self.robot_type] if isinstance(
            q_dsr_reset, dict) else q_dsr_reset
        self.init_pos_bound_dict = init_pos_bound_dict
        self.init_RPY_bound_dict = {k: np.deg2rad(
            v) for k, v in init_RPY_bound_dict.items()}
        self.init_orgin_T = RPY2T(
            *init_orgin_RPY[:3], *radian_list(init_orgin_RPY[3:]))
        # self.is_done_out_ws = is_done_out_ws
        # self.is_done_out_jnt = is_done_out_jnt
        self.is_random_init_pose = is_random_init_pose
        self.is_depth = is_depth
        self.cam_depth_z_min = cam_depth_z_min
        self.cam_depth_z_max = cam_depth_z_max
        self.manual_set_base_rpy = manual_set_base_rpy[:3] + \
            radian_list(manual_set_base_rpy[3:])
        self.gripper_pos_min = gripper_pos_min
        self.gripper_pos_max = np.deg2rad(gripper_pos_max)
        self.depth_image_max = depth_image_max
        self.depth_image_min = depth_image_min
        self.verbose = verbose
        assert ws_x_high > ws_x_low
        assert ws_y_high > ws_y_low
        assert ws_z_high > ws_z_low
        assert action_dx_max > 0, 'positive num'
        assert action_dy_max > 0, 'positive num'
        assert action_dz_max > 0, 'positive num'
        assert cam_width == cam_width, 'only support square'
        assert obs_type == 'image', 'not implement'
        assert obs_cam_type == 'rgb', 'not implement'
        assert obs_cam_device in ['cameraL'], 'not implement'
        assert action_type == 'delta_motion', 'not implement'
        assert action_arm_device in ['psm1', 'psm2'], 'not implement'
        for k, v in self.init_pos_bound_dict.items():
            assert k in ['x', 'y', 'z'], '{k} is not in list'.format(k)
            assert len(v) == 2
            assert v[0] <= v[1]

        for k, v in self.init_RPY_bound_dict.items():
            assert k in ['R', 'P', 'Y'], '{k} is not in list'.format(k)
            assert len(v) == 2
            assert v[0] <= v[1]
        self.metadata = {'render.modes': ['rgb_array']}
        if not is_dummy:
            self._init_ros_client()
            self.T_dsr_current = self.clients[action_arm_device].T_g_w_dsr
        else:
            self.clients = None
        self.seed = seed
        self.timestep = 0
        self.is_clutch_engage = True

    def step(self, action):
        self.timestep += 1
        if self.is_clutch_engage:
            T_g_w_dsr_nxt, gripper_pos = self.action2client_servo(action)
        else:
            T_g_w_dsr_nxt = self.T_dsr_current
        info = self._get_info(T_g_w_dsr_nxt)
        done = len(info['done']) > 0
        reward = None  # reward was set by upper level wrapper
        if not done and self.is_clutch_engage:
            if (not "ws_out" in info['done']) and (not "q_out" in info['done']):
                # print(info['proprio_info'])
                device = self.action_arm_device
                self.clients[device].servo_tool_cp(
                    T_g_w_dsr_nxt, self.action_arm_itpl_num)
                self.clients[device].wait()
                self.clients[device].servo_jaw_jp(
                    gripper_pos, self.action_jaw_itpl_num)
                self.clients[device].wait()
                self.T_dsr_current = T_g_w_dsr_nxt
                sleep(self.extra_delay_time)

        obs = self._get_obs()
        return obs, reward, done, info

    def reset(self, gripper_pos=1):
        jaw_angle_reset = scale(
            gripper_pos, -1, 1, self.gripper_pos_min, self.gripper_pos_max)
        self.clients[self.action_arm_device].reset_pose(
            q_dsr=self.q_dsr_reset, walltime=None, jaw_angle=jaw_angle_reset)
        self.T_dsr_current = self.clients[self.action_arm_device].T_g_w_dsr
        if self.is_random_init_pose:
            _idx = {'x': 0, 'y': 1, 'z': 2}
            _idx_ROT = {'R': 3, 'P': 4, 'Y': 5}
            low = np.array([self.ws_x_low, self.ws_y_low,
                           self.ws_z_low, -np.pi, -np.pi, -np.pi])  # by default
            high = np.array([self.ws_x_high, self.ws_y_high,
                            self.ws_z_high, np.pi, np.pi, np.pi])
            for k, v in self.init_pos_bound_dict.items():
                low[_idx[k]] = v[0]
                high[_idx[k]] = v[1]
            for k, v in self.init_RPY_bound_dict.items():
                low[_idx_ROT[k]] = v[0]
                high[_idx_ROT[k]] = v[1]
            # _mean = (high + low)/2
            # _scale = (high - low)/2

            while True:
                pose = self.rng_pose.uniform(low, high)
                # print(f"low: {low}")
                # print(f"high: {high}")
                # print(pose)

                T = RPY2T(*pose.tolist())
                T = Frame(T.M*self.init_orgin_T.M, self.init_orgin_T.p + T.p)
                # print(f'low: {low} high: {high}, dsr: {T.p}')
                # print(T)
                _q = self.clients[self.action_arm_device].ik_map(T)
                is_out_jnt, _ = self.is_out_q_limits(
                    _q, self.action_arm_device)
                if not is_out_jnt:
                    break
                print("skip out qlim start pose..")
            self.clients[self.action_arm_device].servo_tool_cp(
                T, interpolate_num=50, clear_queue=False)
            self.clients[self.action_arm_device].wait()
            self.clients[self.action_arm_device].servo_jaw_jp(
                jaw_angle_reset, self.action_jaw_itpl_num)
            self.clients[self.action_arm_device].wait()
            self.T_dsr_current = T

        print("finish reset")
        # print("T dsr", np.around(T2RPY(self.unwrapped.clients[self.unwrapped.action_arm_device].T_g_w_dsr)[0], 3))
        # print("T dsr 2", np.around(T2RPY(T)[0], 3))
        # self.clients['ecm'].move_ecm_jp(self.ecm_init_q, time_out=40)
        sleep(1)  # waiting for PID controller stablized
        # print("exit reset")
        obs = self._get_obs()
        self.timestep = 0
        self.is_clutch_engage = True
        # print('reset env', end='\r')
        return obs

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int)
        self._seed = seed
        self.rng_pose = np.random.RandomState(seed)  # use local seed
        self.rng = np.random.RandomState(seed)  # other local seed

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit gym env")
        self.close()

    def open_gripper(self):
        self.clients[self.action_arm_device].servo_jaw_jp(
            self.gripper_pos_max, self.action_jaw_itpl_num)
        self.clients[self.action_arm_device].wait()

    def close(self):
        if self.clients is not None:
            for _, client in self.clients.items():
                client.close()  # close threads in clients
                del client
            self.clients = {}

    def action2client_servo(self, action):
        """ send cmd to low-level controller """
        _action = action.copy(
        )  # important!!! if not copy the variable is like pointer. lead to unstable
        dx, dy, dz = _action[0] * self.action_dx_max, _action[1] * \
            self.action_dy_max, _action[2] * self.action_dz_max
        rotR, rotP, rotY = _action[3] * self.action_dR_max, _action[4] * \
            self.action_dP_max, _action[5] * self.action_dY_max
        # map from [-1 1] to [0 pi/8]
        gripper_pos = scale(action[-1], -1, 1,
                            self.gripper_pos_min, self.gripper_pos_max)

        data = [dx, dy, dz, rotR, rotP, rotY, gripper_pos]
        device = self.action_arm_device

        if self.T_dsr_current is None:
            self.T_dsr_current = self.clients[device].T_g_w_dsr
        deltaT = RPY2T(*data[:-1])
        # Here are a mistake version:  RPY2T(*data[:-1]) * self.engines['psm2'].clients['psm2'].T_g_w_dsr
        T_g_w_dsr_nxt = Frame(deltaT.M * self.T_dsr_current.M,
                              deltaT.p + self.T_dsr_current.p)

        return T_g_w_dsr_nxt, data[-1]

    def render(self, mode, *args, **kwargs):
        if mode == 'rgb_array':
            return self._get_obs()['image']
        elif mode == 'depth_xyz':
            return self._get_obs()['depth_xyz']

    def _init_ros_client(self):
        """ initialize ros clients """

        self.clients = {}
        self._ros_node = None
        self._ros_node = rospy.init_node('ros_client_engine', anonymous=True)

        if self.robot_type == 'ambf':
            self.clients[self.action_arm_device] = PSMClient(self._ros_node,
                                                             self.action_arm_device,
                                                             default_servo_type='servo_jp',
                                                             robot_type=self.robot_type,
                                                             default_kin_engine='surgical_challenge',
                                                             is_measure_is_grasp=True,
                                                             qlim_MarginRatio=self.q_margin_ratio,
                                                             jaw_qlim_MarginRatio=self.q_margin_ratio,
                                                             ignore_qlim=True
                                                             )

            self.clients['ecm'] = ECMClient(self._ros_node,
                                            robot_type=self.robot_type,
                                            is_left_point_cloud=self.is_depth,
                                            is_left_cam=(not self.is_depth))
            self.clients['world'] = WolrdClient(ros_node=self._ros_node,
                                                robot_type=self.robot_type,
                                                )

        elif self.robot_type == 'dvrk_2_0':
            self.clients[self.action_arm_device] = PSMClient(self._ros_node,
                                                             arm_name=self.action_arm_device,
                                                             robot_type='dvrk_2_0',
                                                             default_servo_type='move_jp',
                                                             # default_kin_engine='peter_corke',
                                                             default_kin_engine='surgical_challenge',
                                                             is_measure_is_grasp=True,  # use other method to detect
                                                             qlim_MarginRatio=self.q_margin_ratio,
                                                             jaw_qlim_MarginRatio=self.q_margin_ratio,
                                                             manual_set_base_rpy=self.manual_set_base_rpy,
                                                             ignore_qlim=True,
                                                             )
            self.clients['ecm'] = ECMClient(self._ros_node,
                                            robot_type=self.robot_type,
                                            is_left_point_cloud=self.is_depth,
                                            is_left_cam=(not self.is_depth))
            self.clients['world'] = WolrdClient(
                self._ros_node, robot_type=self.robot_type)

        for _, _client in self.clients.items():
            _client.start()

    def _get_obs(self):
        def keep_dim(x): return x.reshape(
            x.shape + (1,)) if len(x.shape) == 2 else x
        obs = {}
        if self.obs_cam_device == 'cameraL':
            if self.is_depth:
                if self.robot_type == 'ambf':
                    data = self.clients['ecm'].get_signal(
                        'cameraL_point_cloud')
                    _depth_xyz, _img = PointCloud2_2_xyzNimage(
                        data, height=480, width=640)  # size need to find out from topics
                    _depth_xyz = convert_xyz_2_depth_image(
                        _depth_xyz,  z_min=self.cam_depth_z_min, z_max=self.cam_depth_z_max, is_reverse=False)
                    assert _depth_xyz.shape[0] == _img.shape[0]
                    assert _depth_xyz.shape[1] == _img.shape[1]
                else:
                    _depth_xyz = self.clients['ecm'].get_signal(
                        'cameraL_depth_image')
                    assert _depth_xyz is not None
                    # scale uint16 to uint8, refer to solution: https://stackoverflow.com/questions/11337499/how-to-convert-an-image-from-np-uint16-to-np-uint8
                    _depth_xyz = scale_rawdepth_2_uint8depth(
                        _depth_xyz, z_min=self.cam_depth_z_min, z_max=self.cam_depth_z_max, image_min=self.depth_image_min, image_max=self.depth_image_max)
                    _depth_xyz = keep_dim(_depth_xyz)
                    _img = self.clients['ecm'].get_signal('cameraL_image')
            else:
                _img = self.clients['ecm'].get_signal('cameraL_image')
                _depth_xyz = None
        else:
            raise NotImplementedError

        def crop_ratio(x): return x[int(x.shape[0]*self.cam_height_crop[0]):int(x.shape[0]*self.cam_height_crop[1]),
                                    int(x.shape[1]*self.cam_width_crop[0]):int(x.shape[1]*self.cam_width_crop[1]),
                                    :]
        _img = crop_ratio(_img)
        _img = resize_img(crop2square_img(_img), self.cam_width)
        obs['image'] = _img

        if _depth_xyz is not None:
            _depth_xyz = keep_dim(crop_ratio(_depth_xyz))
            _depth_xyz = keep_dim(resize_img(
                keep_dim(crop2square_img(_depth_xyz)), self.cam_width))
            obs['depth_xyz'] = _depth_xyz
        return obs

    def _get_info(self, T_g_w_dsr):
        # device = self.action_arm_device
        # T_g_w_dsr = self.clients[device].T_g_w_dsr
        # qs = self.clients[device]._q_dsr

        device = self.action_arm_device
        qs = self.clients[device].ik_map(T_g_w_dsr)
        is_out_jnt, is_out_jnt_result = self.is_out_q_limits(qs, device)
        info = {'done': [], 'proprio_info': []}

        # workspace
        if self.is_out_workspace(T_g_w_dsr.p.x(), T_g_w_dsr.p.y(), T_g_w_dsr.p.z()):
            _info = {'ws_out':
                     {'gripper_pos': [T_g_w_dsr.p.x(), T_g_w_dsr.p.y(), T_g_w_dsr.p.z()],
                      'ws_low': [self.ws_x_low, self.ws_y_low, self.ws_z_low],
                      'ws_high': [self.ws_x_high, self.ws_y_high, self.ws_z_high], }
                     }
            info['proprio_info'].append('ws_out')
            # print("Desire Out Ws")
            info['done'].append(_info)

        # joint space
        elif is_out_jnt:
            _info = {'q_out':
                     {'result': is_out_jnt_result,
                         'q': qs}
                     }
            # print("Desire Out Joint Limit")
            info['proprio_info'].append('q_out')
            info['done'].append(_info)

        info['is_clutch_engage'] = self.is_clutch_engage

        return info

    def is_out_workspace(self, x, y, z):
        return not (self.ws_x_low < x and self.ws_x_high > x and
                    self.ws_y_low < y and self.ws_y_high > y and
                    self.ws_z_low < z and self.ws_z_high > z)

    def is_out_q_limits(self, qs, device_name):
        return self.clients[device_name].is_out_qlim(qs, margin_ratio=self.q_margin_ratio)

    def get_oracle_action(self, obs):
        raise NotImplementedError

    @property
    def action_space(self):
        # action normalized to range -1~1
        return spaces.Box(low=np.float32(-1*np.ones(7)), high=np.float32(1*np.ones(7)), dtype=np.float32)

    @property
    def observation_space(self):
        """ 
        refer to
        https://github.com/jsikyoon/dreamer-torch/blob/main/wrappers.py
        https://github.com/ryanhoque/fabric-vsf/blob/master/gym_cloth/envs/cloth_env.py
        """
        obs = {}
        obs['image'] = spaces.Box(0, 255, (self.cam_width, self.cam_height, 3),
                                  dtype=np.uint8)
        return gym.spaces.Dict(obs)
