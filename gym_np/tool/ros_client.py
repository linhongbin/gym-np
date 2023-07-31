## ros related
from sensor_msgs.msg import JointState, Image, PointCloud2, CompressedImage
from geometry_msgs.msg import TransformStamped, Transform, TwistStamped, PoseStamped
from rospy import Publisher, Subscriber, Rate, init_node, spin, get_published_topics, Rate, is_shutdown
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Bool, Empty

## common
from typing import List, Any, Dict, Optional
from time import sleep
from PyKDL import Frame, Rotation, Vector
from dataclasses import dataclass
import numpy as np
from time import sleep
from threading import Thread
import time
from queue import Queue
from typing import List
from simple_pid import PID

### gym_np related
from gym_np.tool.common import Quaternion2T, RPY2T, gen_interpolate_frames, SE3_2_T, T_2_SE3, RigidBodyState2T, PoseStamped2T, T2Quaternion
from gym_np.tool.kinematics import PSM_KIN
from gym_np.script.surgical_challenge_api_mod.psmIK_mod import compute_IK
from gym_np.script.surgical_challenge_api_mod.psmFK_mod import compute_FK
from surgical_robotics_challenge.utils.utilities import convert_mat_to_frame

try:
    from ambf_msgs.msg import RigidBodyState, RigidBodyCmd, CameraCmd
    IS_AMBF_LOADED = True
except:
    IS_AMBF_LOADED = False

def T2TransformStamped(T):
    """ 
    Frame to Ros Message:TransformStamped
    """
    msg = TransformStamped()
    rx,ry,rz,rw, = T.M.GetQuaternion()
    msg.transform.rotation.x = rx
    msg.transform.rotation.y = ry
    msg.transform.rotation.z = rz
    msg.transform.rotation.w = rw
    msg.transform.translation.x = T.p.x()
    msg.transform.translation.y = T.p.y()
    msg.transform.translation.z = T.p.z()
    return msg  

def TransformStamped2T(msg):
    """ 
    Ros Message:TransformStamped to Frame
    """
            
    x= msg.transform.translation.x
    y= msg.transform.translation.y
    z= msg.transform.translation.z

    Qx = msg.transform.rotation.x
    Qy = msg.transform.rotation.y
    Qz = msg.transform.rotation.z
    Qw = msg.transform.rotation.w
    T = Quaternion2T(x, y, z, Qx,Qy,Qz,Qw)
    return T  

if IS_AMBF_LOADED:
    def RigidBodyState2T(msg):
        """
        RigidBodyState message to Frame 
        """
        
        x= msg.pose.position.x
        y= msg.pose.position.y
        z= msg.pose.position.z

        Qx = msg.pose.orientation.x
        Qy = msg.pose.orientation.y
        Qz = msg.pose.orientation.z
        Qw = msg.pose.orientation.w
        T = Quaternion2T(x, y, z, Qx,Qy,Qz,Qw)
        return T
    
    def T2RigidBodyCmd(T):
        """
        RigidBodyState message to Frame 
        """
        msg = RigidBodyCmd()
        msg.cartesian_cmd_type = 1
        quat = T2Quaternion(T)
        msg.pose.position.x = quat[0]
        msg.pose.position.y = quat[1]
        msg.pose.position.z = quat[2]
        msg.pose.orientation.x = quat[3]
        msg.pose.orientation.y = quat[4]
        msg.pose.orientation.z = quat[5]
        msg.pose.orientation.w = quat[6]
        return msg
    
    def T2CameraCmd(T):
        """
        RigidBodyState message to Frame 
        """
        msg = CameraCmd()
        msg.enable_position_controller = True
        quat = T2Quaternion(T)
        msg.pose.position.x = quat[0]
        msg.pose.position.y = quat[1]
        msg.pose.position.z = quat[2]
        msg.pose.orientation.x = quat[3]
        msg.pose.orientation.y = quat[4]
        msg.pose.orientation.z = quat[5]
        msg.pose.orientation.w = quat[6]
        return msg


class ClientEngine():
    """
    an engine to manage clients that subscribe and publish CRTK related topics.
    """
    ROS_RATE = 100
    def __init__(self):
        self.clients = {}
        self.ros_node = init_node('ros_client_engine',anonymous=True)
        self.ros_rate = Rate(self.ROS_RATE)


    def add_clients(self, client_names:List[str], robot_type='ambf'):
        for client_name in client_names:
            self._add_client(client_name, robot_type)
        sleep(1) # wait for ros subscribed topics to be ok
    
    def add_client_obj(self, obj_name, obj):
        self.clients[obj_name] = obj

    def start(self):
        for client in self.clients.values():
            client.start()
    
    def close(self):
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                print(str(e))
    
    def close_client(self, name):
         self.clients[name].close()
         del self.clients[name]

    def get_signal(self, client_name, signal_name):
        """
        get signal data from client
        """
        self._is_has_client(client_name, raise_error=True)
        return self.clients[client_name].get_signal(signal_name)

    def _add_client(self, client_name, robot_type='ambf'):
        if client_name in ['psm1', 'psm2']:
            self.clients[client_name] = PSMClient(self.ros_node, client_name, robot_type)
        elif client_name == 'ecm':
            self.clients[client_name] = ECMClient(self.ros_node)
        elif client_name == 'scene':
            self.clients[client_name] = SceneClient(self.ros_node)
        elif client_name == 'ambf':
            self.clients[client_name] = WolrdClient(self.ros_node)
        else:
            raise NotImplementedError

    def _is_has_client(self,client_name, raise_error=False):
        result = client_name in list(self.clients.keys())
        if raise_error and not result:
            raise Exception(f"client {client_name} has not been added")
        
    @property
    def client_names(self):
        return list(self.clients.keys())
    
@dataclass
class SignalData:
    """ data class for testing if signal is died"""
    data:Any
    counter:int


class BaseClient():
    """ 
    define some common property for clients 
    """
    COUNTER_BUFFER = 500
    def __init__(self, ros_node, is_run_thread=False, robot_type="ambf"):
        self.ros_node =ros_node
        self.sub_signals = {}
        self.subs = dict()
        self.pubs = dict()
        self.is_alive = False
        self.robot_type=robot_type

        self.is_run_thread = is_run_thread
        if self.is_run_thread:
            self.thread = Thread(target=self.run) 
        else:
            self.thread = None

    def start(self):
        self.is_alive = True
        if self.is_run_thread:
            self.thread.start()
    def close(self):
        self.is_alive = False


    def topic_wrap(self, topic_name, raise_exception=True):
        """ wrapper for ros topic name, check if the topic name exists
        
        raise exception if not exists
        """
        topics = [topic[0] for topic in get_published_topics()]
        result = topic_name in topics
        if (not result) and raise_exception:
            print(topics)
            raise Exception("topic {} does not exist, please check if crtk interface is running".format(topic_name))
        return topic_name
    
    
    def is_has_signal(self, signal_name:str):
        """
        check if signal name exists
        """
        return signal_name in self.sub_signals_names

    def set_signal(self, signal_name:str, data):
        """ 
        update signal data 
        """
        if not self.is_has_signal(signal_name):
            self.sub_signals[signal_name] = SignalData(data= data, counter=1)
        else:
            cnt = self.sub_signals[signal_name].counter
            cnt = cnt +1 if cnt<=self.COUNTER_BUFFER else self.COUNTER_BUFFER
            self.sub_signals[signal_name] = SignalData(data = data, counter = cnt)
    
    def get_signal(self, signal_name:str):
        """
        get signal data
        """
        if not self.is_has_signal(signal_name):
            # print(f"{signal_name} is not exist")
            return None
        else:
            return self.sub_signals[signal_name].data

    def reset_signal_counter(self, signal_name:str):
        """
        set signal coutner to 0
        """
        if self.is_has_signal(signal_name):
            self.sub_signals[signal_name].counter = 0

    def run():
        raise NotImplementedError
    
    @property
    def sub_signals_names(self):
        return list(self.sub_signals.keys())
    @property
    def sub_topics_names(self):
        return list(self.subs.keys())
    @property
    def pub_topics_names(self):
        return list(self.pubs.keys())

class WolrdClient(BaseClient):
    def __init__(self, 
                 ros_node,
                 robot_type:str,
                 verbose=0):
        super(WolrdClient, self).__init__(ros_node, 
                                         is_run_thread=False,)
        self.robot_type = robot_type
        self.verbose=verbose
        
        if self.robot_type == "ambf":
            self.pubs["wolrd_reset"] = Publisher({"ambf": '/ambf/env/World/Command/Reset',}
                                            [self.robot_type], Empty, queue_size=1)
            self.pubs["wolrd_reset_bodies"] = Publisher({"ambf": '/ambf/env/World/Command/Reset/Bodies',}
                                            [self.robot_type], Empty, queue_size=1) 
            self.pubs["needle"] = Publisher({"ambf": '/ambf/env/Needle/Command',}
                                            [self.robot_type], RigidBodyCmd, queue_size=1)    
            self.pubs["camera"] = Publisher({"ambf": '/ambf/env/cameras/cameraL/Command',}
                                            [self.robot_type], CameraCmd, queue_size=1) 
            self.subs["needle_pose"] = Subscriber({"ambf": '/ambf/env/Needle/State',}
                                                [self.robot_type], RigidBodyState, self._measured_needle_pose_cb)
            self.rate = Rate(100)

    def reset_all(self):
        """ reset all, similar to ctrl+R 
        """
        if self.robot_type == 'ambf':
            msg = Empty()
            self.pubs["wolrd_reset"].publish(msg) # Resets the whole world (Lights, Cams, Volumes, Rigid Bodies, Plugins etc)
            time.sleep(0.5)
            msg = Empty()
            self.pubs["wolrd_reset_bodies"].publish(msg) # Reset Static / Dynamic Rigid Bodies
        elif self.robot_type == 'dvrk_2_0':
            if self.verbose: print("require manually reset scene..")
    
    def reset_bodies(self):
        if self.robot_type == 'ambf':
            msg = Empty()
            self.pubs["wolrd_reset_bodies"].publish(msg) # Reset Static / Dynamic Rigid Bodies
        elif self.robot_type == 'dvrk_2_0':
            if self.verbose: print("require manually reset scene..")

    # def reset_needle(self, accuracy=0.09):
    #     """ reset needle to starting position 
    #     """
    #     if self.robot_type == 'ambf':
    #         pos = [-0.20786757338201337, 0.5619611862776279, 0.7317253877244148] # hardcode this position
    #         rpy = [0.03031654271074325, 0.029994510295635185, -0.00018838556827461113]
    #         # pos_ready = pos.copy()
    #         # pos_ready[1] = pos_ready[1] - 0.3
    #         # pos_ready[2] = pos_ready[2] + 0.2
    #         repeat = 2
    #         for i in range(10):
    #             if i+1 == repeat:
    #                 self.reset_all()
    #             else:
    #                 self.set_needle_pose(pos=pos, rpy=rpy)
    #             _pos, _rpy = self.get_needle_pose()
    #             err = np.linalg.norm(np.array(_pos)-np.array(pos))
    #             if err<accuracy: # ensure needle stay within target
    #                 break
    #     elif self.robot_type == 'dvrk_2_0':
    #         if self.verbose:  print("require manually reset needle..")

    def set_needle_pose(self, pos:List[float]=None, rpy:List[float]=None):
        """ set needle position 
        """
        if self.robot_type == 'ambf':
            assert not ((pos is None)and(rpy is None)), 'pos and rpy cannot be None in the same time'
            if pos is None:
                pos, _ = self.get_needle_pose()
            if rpy is None:
                _, rpy = self.get_needle_pose()
            
            msg = T2RigidBodyCmd(RPY2T(*pos,*rpy))
            for i in range(10):
                self.pubs["needle"].publish(msg)
                self.rate.sleep()
            # for _ in range(3):
            #     needle.set_force(0, 0, 0)
            #     needle.set_torque(0, 0, 0)
            time.sleep(1)
        elif self.robot_type == 'dvrk_2_0':
            if self.verbose:  print("require manually set needle..")
            
    def servo_cam_local(self, pos, quat, cam='cameraL', is_fixed_cam=False):
        """ servo camera w.r.t. ECM base frame 
        """
        if self.robot_type == 'ambf':
            assert cam in ['cameraL'], 'not implement'
            # cam = self.client.get_obj_handle('/ambf/env/cameras/cameraL')
            # rpy = Rotation.Quaternion(*quat).GetRPY()
            # cam.set_pos(*pos)
            # cam.set_rpy(*rpy)
            msg = T2CameraCmd(Quaternion2T(*pos, *quat))
            for i in range(10):
                self.pubs["camera"].publish(msg)
                self.rate.sleep()
            
            if not is_fixed_cam:
                msg = CameraCmd()
                msg.enable_position_controller = False
                for i in range(10):
                    self.pubs["camera"].publish(msg)
                    self.rate.sleep()
                
        elif self.robot_type == 'dvrk_2_0':
            if self.verbose:  print("require manually servo camera..")
        
    def get_needle_pose(self):
        if self.robot_type == 'ambf':
            T = self.get_signal('T_needle_msr')
            assert T is not None
            pos = [T.p.x(),T.p.y(),T.p.z()]
            rpy = list(T.M.GetRPY())
            return pos, rpy
        elif self.robot_type == 'dvrk_2_0':
            if self.verbose:  print("require manually get needle pose.., return dummpy T")
            return [0,0,0], [0,0,0]
        
    def _measured_needle_pose_cb(self, data):
        self.set_signal('T_needle_msr',RigidBodyState2T(data))
 

class PSMClient(BaseClient):
    def __init__(self, ros_node,
                       arm_name:str,
                       robot_type:str,
                       default_servo_type="servo_jp",
                       IK_MAX_NUM:int= 2,
                       reset_jp: List[float] = {'ambf':
                                                    {'psm1':[0.2574930501388846, -0.2637599054454396, 1.490778072887017, -2.3705447576048746, 0.3589815573742414, -1.0241148122485695],
                                                    'psm2':[-0.5656515955924988, -0.15630173683166504, 1.3160043954849243, -2.2147457599639893, 0.8174221515655518,-1]},
                                                'dvrk_2_0':
                                                    {'psm1':[0,0,0.12,0,0,0],
                                                    'psm2':[0,0,0.12,0,0,0]},
                                                        
                                                },
                       pids_param_dict:Dict[str, List[List[float]]] = {}, 
                       grasp_point_offset_rpy={'ambf':[0, 0, -0.035, 0,0,0],
                                               'dvrk_2_0':[0, 0, 0, 0,0,0],},
                       default_kin_engine='surgical_challenge',
                       TrackErrorMax = {'ambf':[0.8, 0.8, 0.05, 0.8,0.8,0.8], # ambf distance is 10x scale, angle remains the same
                                        'dvrk_2_0':[0.8, 0.8, 0.005, 0.8,0.8,0.8]}, # the 3rd element correspond to prismatic joint,
                       JawTrackErrorMax = {'ambf':0.8, # ambf is 10x scale
                                           'dvrk_2_0':0.8,},
                       qlim_MarginRatio=0.1,
                       jaw_qlim_MarginRatio=0.1,
                       is_measure_is_grasp = False,
                       ignore_tracking_error= True,
                       ignore_qlim=True,
                       manual_set_base_rpy=None,
                        ):
        super(PSMClient, self).__init__(ros_node, 
                                        is_run_thread=True,
                                        robot_type=robot_type)
        
        
        #=== args sanity check
        assert arm_name  in ['psm1', 'psm2'], f"{arm_name}"
        
        
        self.default_servo_type = default_servo_type
        self.servo_type = default_servo_type
        self.IK_MAX_NUM = IK_MAX_NUM
        self.reset_jp = reset_jp[robot_type]
        self.arm_name = arm_name
        self.grasp_point_offset = RPY2T(*grasp_point_offset_rpy[robot_type])
        self.default_kin_engine = default_kin_engine
        self.TrackErrorMax = TrackErrorMax[robot_type]
        self.qlim_MarginRatio = qlim_MarginRatio
        self.JawTrackErrorMax = JawTrackErrorMax[robot_type]
        self.jaw_qlim_MarginRatio = jaw_qlim_MarginRatio
        self.available_servo_types = ["servo_jp", "move_jp"]
        assert default_servo_type in self.available_servo_types
        
        #========= init variables
        self.kin = PSM_KIN(robot_type) # kinematics model
        self._jaw_pub_queue = Queue()
        self._arm_pub_queue = Queue()
        self._q_dsr =None
        self._jaw_dsr = None
        self.joint_calibrate_offset = np.array([0,0,0, 0,0,0])
        self._ignore_tracking_error =ignore_tracking_error
        self._ignore_qlim = ignore_qlim
        self.is_measure_is_grasp = is_measure_is_grasp

        ##============ ros pub topics
        self.pubs["servo_jp"] = Publisher({"ambf": '/CRTK/'+ arm_name +'/servo_jp',
                                           "dvrk_2_0": '/'+arm_name.upper() +'/servo_jp'}
                                           [self.robot_type], JointState, queue_size=1)
        self.pubs["servo_jaw_jp"] = Publisher({"ambf": '/CRTK/'+ arm_name +'/jaw/servo_jp',
                                                "dvrk_2_0":'/'+arm_name.upper() +'/jaw/servo_jp'} # servo cannot move jaw and arm at the same time in dVRK2.0                                              
                                                    [self.robot_type], JointState, queue_size=1)
        
        self.pubs["move_jp"] = Publisher({"ambf": '/CRTK/'+ arm_name +'/move_jp',
                                           "dvrk_2_0": '/'+arm_name.upper() +'/move_jp'}
                                           [self.robot_type], JointState, queue_size=1)
        self.pubs["move_jaw_jp"] = Publisher({"ambf": '/CRTK/'+ arm_name +'/jaw/move_jp',
                                                "dvrk_2_0":'/'+arm_name.upper() +'/jaw/move_jp'}                                              
                                                    [self.robot_type], JointState, queue_size=1)
        
        ##============ ros sub topics
        self.subs["measured_js"] = Subscriber({"ambf": '/CRTK/'+arm_name+'/measured_js', 
                                                "dvrk_2_0":'/'+arm_name.upper() + '/measured_js'}
                                                [self.robot_type], JointState, self._measured_js_cb)
        self.subs["measured_jaw_js"] = Subscriber({"ambf": '/CRTK/'+arm_name+'/measured_jaw_js', 
                                                "dvrk_2_0":'/'+arm_name.upper() + '/jaw/measured_js'} 
                                                [self.robot_type], JointState, self._measured__jaw_js_cb)

        if self.robot_type == 'ambf':
            if manual_set_base_rpy is None:
                self.subs["measured_base_cp"] = Subscriber(self.topic_wrap('/ambf/env/'+arm_name+'/baselink/State'), RigidBodyState, self._measured_base_cp_cb)
            else:
                assert len(manual_set_base_rpy) == 6
                self.set_signal('measured_base_cp', RPY2T(*manual_set_base_rpy))
            if is_measure_is_grasp:
                self.subs["measured_is_grasp"] = Subscriber(self.topic_wrap('/CRTK/'+ arm_name +'/is_grasp'), Bool, self._measured_is_grasp)
            self.ros_rate = Rate(120) 
        elif self.robot_type == 'dvrk_2_0':
            # crtk related
            from crtk_msgs.msg import operating_state
            assert manual_set_base_rpy is not None, "dvrk should manually set base pose"
            assert len(manual_set_base_rpy) == 6
            self.set_signal('measured_base_cp', RPY2T(*manual_set_base_rpy)) # by default, need to manually set if it is changed
            self.subs["operating_state"] = Subscriber('/'+arm_name.upper() + '/operating_state', 
                                                      operating_state, self._operating_state_cb)
            self.ros_rate = Rate(120)
            
# operating_state
        
        #======= update pid controller
        self.update_pid(pids_param_dict)
        
        while (self._q_dsr is None) or (self._jaw_dsr is None):
            self.reset_dsr_to_msr()
            sleep(0.1)
            print("wait for signals..", "q_dsr:", self._q_dsr, "jaw_dsr:", self._jaw_dsr )
        


    def update_pid(self,pids_param_dict):
        """Update PID controllers
        """
        self.pids_param_dict = pids_param_dict
        self.pids = {}
        for k, v in pids_param_dict.items():   
            self.pids[k] = [PID(v[0,i], 
                                v[1,i], 
                                v[2,i], 
                            setpoint=0, 
                            output_limits=(-self.PID_Clamp, self.PID_Clamp),
                            sample_time=0.01) for i in range(6)]

    def reset_pose(self, q_dsr:List=None, walltime=None, jaw_angle=0, interpolate_num=100):
        """move to reset joint position
        """

        _q_dsr = q_dsr or self.reset_jp[self.arm_name]
        # self._ignore_qlim =True
        _ignore_tracking_error = self._ignore_tracking_error
        self._ignore_tracking_error =True
        # self.servo_jp(self._q_dsr, interpolate_num=None)  
        # self.wait(walltime=walltime)
        self.servo_jp(_q_dsr, interpolate_num=interpolate_num)
        # for _ in range(400):
        #     self.servo_jp(_q_dsr, interpolate_num=None)
        self.open_jaw(jaw_angle)
        self.wait(walltime=walltime)
        sleep(0.1)
        self.reset_dsr_to_msr()
        # self._ignore_qlim =False
        self._ignore_tracking_error =_ignore_tracking_error

    
    def wait(self, walltime=None, force_walltime=False, is_busy=False):
        """ wait until the queues are empty
        """
        assert not ((walltime is None) and force_walltime), 'need to assert walltime when force_walltime is True'
        start = time.time()
        while True:
            if self.servo_type == 'servo_jp':
                if not walltime is None:
                    if (time.time()-start)>walltime:
                        self._arm_pub_queue.queue.clear()
                        self._jaw_pub_queue.queue.clear()
                        break
                if self._arm_pub_queue.empty() and self._jaw_pub_queue.empty() and (not force_walltime):
                    break
            elif self.servo_type == 'move_jp':
                if is_busy :
                    if self.get_signal("is_busy") or (time.time()-start)>0.2:
                        break
                elif self._arm_pub_queue.empty() and self._jaw_pub_queue.empty() and (not self.get_signal("is_busy")) and (time.time()-start)>0.2:
                    break
            self.ros_rate.sleep()

    def servo_tool_cp_local(self, T_g_b_dsr:Frame,interpolate_num=None, clear_queue=True,servo_type:str=None):
        """servo tool cartesian pose from PSM base to tip
        """
        self.servo_type = servo_type or self.default_servo_type
        assert self.servo_type in self.available_servo_types
        
        T0 = self.T_g_b_dsr
        if clear_queue:
            self._arm_pub_queue.queue.clear()
        if interpolate_num is None or self.servo_type=='move_jp':
            _T_t_b_dsr = T_g_b_dsr * self.grasp_point_offset.Inverse()
            self._arm_pub_queue.put(_T_t_b_dsr)
        else:
            frames = gen_interpolate_frames(T0, T_g_b_dsr, interpolate_num)
            for i, frame in enumerate(frames):
                _T_t_b_dsr = frame * self.grasp_point_offset.Inverse()
                self._arm_pub_queue.put(_T_t_b_dsr)
    
    def servo_tool_cp(self,  T_g_w_dsr:Frame, interpolate_num=None, clear_queue=True, servo_type:str=None):
        """servo tool cartesian pose from world base to PSM base

        Args:
            T_g_w_dsr (Frame): _description_
            interpolate_num (_type_, optional): _description_. Defaults to None.
            clear_queue (bool, optional): _description_. Defaults to True.
        """
        T_b_w = self.get_signal('measured_base_cp')
        T_g_b_dsr = T_b_w.Inverse()*T_g_w_dsr 
        self.servo_tool_cp_local(T_g_b_dsr, interpolate_num, clear_queue, servo_type)


    def servo_jaw_jp(self, jaw_jp_dsr:float, interpolate_num=None,clear_queue=True, servo_type:str=None):
        """move jaw joint position
        """
        self.servo_type = servo_type or self.default_servo_type
        assert self.servo_type in self.available_servo_types

        if clear_queue:
            self._jaw_pub_queue.queue.clear()
        if interpolate_num is None or self.servo_type=='move_jp':
            self._jaw_pub_queue.put(jaw_jp_dsr)
        else:
            qs = np.linspace(self.jaw_dsr, jaw_jp_dsr, interpolate_num).tolist()
            for q in qs:
                _jaw_jp_dsr = q
                self._jaw_pub_queue.put(_jaw_jp_dsr)

    def servo_jp(self, q_dsr:List, interpolate_num=None,clear_queue=True, servo_type:str=None):
        """servo to desired joint position
is_out_qlimlear_queue (bool, optional): _description_. Defaults to True.
        """
        self.servo_type = servo_type or self.default_servo_type
        assert self.servo_type in self.available_servo_types
        q0 = self._q_dsr
        if clear_queue:
            self._arm_pub_queue.queue.clear()
        if interpolate_num is None or self.servo_type=='move_jp':
            self._arm_pub_queue.put(q_dsr)
        else:
            qs = np.linspace(q0, q_dsr, interpolate_num).tolist()
            for q in qs:
                self._arm_pub_queue.put(q)

    def close_jaw(self):
        self.servo_jaw_jp(0, 200)

                
    def open_jaw(self, angle=None):
        if angle is None:
            self.servo_jaw_jp(0.4, 100)
        else:
            self.servo_jaw_jp(angle, 100)
            
    
    def fk_local(self, qs:List[float], engine:Optional[str]=None):
        _engine = engine or self.default_kin_engine
        if _engine == 'peter_corke':       
            T = SE3_2_T(self.kin.fk(qs))
        
        elif _engine == 'surgical_challenge':
            T = convert_mat_to_frame(compute_FK(qs, 7, self.robot_type))
        return T
    def fk(self, qs:List[float],  engine:Optional[str]=None):
        T_base = self.get_signal('measured_base_cp')
        T = T_base * self.fk_local(qs, engine)
        return T
            
    def ik_local(self, T_g_b:Frame, q0:List=None,  engine:Optional[str]=None):
        """Inverse kinematics from PSM base to tip

        Args:
            T_g_b (Frame): _description_
            q0 (List, optional): _description_. Defaults to None.
            ik_engine (str, optional): _description_. Defaults to 'surgical_challenge'.

        Returns:
            q_dsr, is_success
        """
        _engine = engine or self.default_kin_engine
        if _engine == 'peter_corke':
            if q0 is None:
                q0 = self.get_signal('measured_js')
            q_dsr, is_success = self.kin.ik(T_2_SE3(T_g_b), q0)
        
        elif _engine == 'surgical_challenge':
            q_dsr = compute_IK(T_g_b, self.robot_type)
            is_success = True
        return q_dsr, is_success

    def ik(self, T_g_w:Frame, q0:List=None, engine:Optional[str]=None):
        """Inverse kinematics from world origin to PSM tip

        Args:
            T_g_w (Frame): _description_
            q0 (List, optional): _description_. Defaults to None.

        Returns:
            q_dsr, is_success
        """
        T_b_w = self.get_signal('measured_base_cp')
        T_g_b = T_b_w.Inverse()*T_g_w 
        q_dsr, is_success = self.ik_local(T_g_b, q0, engine)
        return q_dsr, is_success
    
    def reset_dsr_to_msr(self):
        self._q_dsr = self.get_signal('measured_js')
        self._jaw_dsr = self.get_signal('measured_jaw_js')
        
    def is_out_qlim(self, qs, q_min_margin=None, q_max_margin=None, margin_ratio=None):
        out_sum, result = self.kin.is_out_qlim(qs, q_min_margin=q_min_margin, 
                             q_max_margin=q_max_margin, 
                             margin_ratio=margin_ratio)
        if self._ignore_qlim:
            return 0, False
        else:
            return out_sum, result
    
    def is_out_jaw_qlim(self, q, margin_ratio):
        if self._ignore_qlim:
            return False
        else:
            return self.kin.is_out_jaw_qlim(q, margin_ratio)
        
    def run(self):
        """ loop inside a thread 
        """
        self._print_start =  time.time()
        while not is_shutdown() and self.is_alive:
            if self.robot_type == "dvrk_2_0" and self.is_measure_is_grasp:
                self._dvrk_measured_is_grasp_logic()

            #======== joint position servo
            _is_servo_jp = False
            if not self._arm_pub_queue.empty():
                data = self._arm_pub_queue.get()
                if not isinstance(data, list):
                    q0 = self.get_signal('measured_js')
                    q_dsr = None
                    for _ in range(self.IK_MAX_NUM):
                        q_dsr, is_success = self.ik_local(data, q0)
                        if not is_success:
                            q_dsr = None
                        else:
                            break
                else:
                    assert isinstance(data, list)
                    q_dsr = data
                if q_dsr is not None:
                    self._q_dsr = q_dsr
            
                if 'upper_servo_jp' in self.pids:
                    q_msr = self.get_signal('measured_js')
                    e = np.array(self._q_dsr) - np.array(q_msr)
                    q_delta = [-self.pids[i](e[i]) for i in range(6)]
                    q_delta = np.array(q_delta)
                else:
                    q_delta = np.zeros(self.joint_calibrate_offset.shape)

                q_dsr_servo = np.array(self._q_dsr) + q_delta - self.joint_calibrate_offset
                tracking_error = np.abs(q_dsr_servo - np.array(self.get_signal('measured_js')))
                
                
                _error_flag =False
                _is_print =False
                if time.time()- self._print_start>1:
                    self._print_start = time.time()
                    _is_print =True
                if np.sum(tracking_error>self.TrackErrorMax)>0 and \
                            not self._ignore_tracking_error and \
                            not self.servo_type=='move_jp': # skip track error for move_jp
                    _error_flag = True
                    if _is_print:
                        print("====================")
                        print("exceed tracking error: ", tracking_error>self.TrackErrorMax, 
                                "tracking_error:", tracking_error,"tracking_max:", self.TrackErrorMax)
                        
                if self.is_out_qlim(self._q_dsr, margin_ratio=self.qlim_MarginRatio)[0] and not self._ignore_qlim:
                    _error_flag = True
                    if _is_print:
                        print("exceed qlim: ", self.is_out_qlim(self._q_dsr, margin_ratio=self.qlim_MarginRatio)[1], 
                                "qmin:", self.kin.qmin, "qmax", self.kin.qmax, "qs", self._q_dsr)
                if not _error_flag:
                    if self.servo_type == 'servo_jp':
                        msg = JointState()
                        msg.position = q_dsr_servo.tolist()
                        self.pubs['servo_jp'].publish(msg)
                        # print(msg.position)
                        _is_servo_jp = True
                    elif self.servo_type == 'move_jp':
                        msg = JointState()
                        msg.position = q_dsr_servo.tolist()
                        self.pubs['move_jp'].publish(msg)                       
               
            #=============== jaw servo
            if not self._jaw_pub_queue.empty():
                data = self._jaw_pub_queue.get()
                self._jaw_dsr = data
                
                tracking_error = np.abs(self.get_signal("measured_jaw_js") - self._jaw_dsr)
                _error_flag =False
                if np.sum(tracking_error>self.JawTrackErrorMax)>0 and \
                                    not self._ignore_tracking_error and\
                                    not self.servo_type=='move_jp':
                    _error_flag = True
                    if _is_print:
                        print("====================")
                        print("jaw exceed tracking error: ", tracking_error, self.JawTrackErrorMax, "dsr:",self._jaw_dsr, 'msr:', np.abs(self.get_signal("measured_jaw_js")))
                if self.is_out_jaw_qlim(self._jaw_dsr, margin_ratio=self.jaw_qlim_MarginRatio) and not self._ignore_qlim:
                    _error_flag = True
                    if _is_print:
                        print("jaw exceed limit: ",self._jaw_dsr)     
                if not _error_flag:
                    if self.servo_type == 'servo_jp': 
                        if not _is_servo_jp or self.robot_type=='ambf':
                            msg = JointState()
                            msg.position = [self._jaw_dsr]
                            self.pubs["servo_jaw_jp"].publish(msg)
                        else:
                            print("cannot servo jaw and arm at the same time !!")   
                    elif self.servo_type == 'move_jp':    
                        msg = JointState()
                        msg.position = [self._jaw_dsr] 
                        self.pubs["move_jaw_jp"].publish(msg)        

            self.ros_rate.sleep()

    @property
    def jaw_dsr(self):
        return self._jaw_dsr
    @property
    def q_dsr(self):
        return self._q_dsr 
    
    @property
    def T_g_b_dsr(self):
        """ Grasp point frame w.r.t. base, desire"""
        return self.fk_local(self._q_dsr) * self.grasp_point_offset
    @property
    def T_g_w_dsr(self):
        """ Grasp point frame w.r.t. world, desire"""
        return self.get_signal('measured_base_cp') * self.T_g_b_dsr  

    @property
    def T_g_b_msr(self):
        """ Grasp point frame w.r.t. base, measure"""
        _q = self.get_signal('measured_js')
        return self.fk_local(_q) * self.grasp_point_offset
    @property
    def T_g_w_msr(self):
        """ Grasp point frame w.r.t. world, measure"""
        return self.get_signal('measured_base_cp') * self.T_g_b_msr        

    def _measured_base_cp_cb(self, data):
        self.set_signal('measured_base_cp', RigidBodyState2T(data))

    def _measured_js_cb(self, data):
        pos = data.position
        _pos = np.array(pos) + self.joint_calibrate_offset
        self.set_signal('measured_js', _pos.tolist())
        
    def _measured__jaw_js_cb(self, data):
        pos = data.position[0]
        self.set_signal('measured_jaw_js',pos)
    
    def _operating_state_cb(self, data):
        self.set_signal('is_busy',data.is_busy)
    def _measured_is_grasp(self, data):
        self.set_signal('measured_is_grasp', data.data)

    def fk_map(self, q):
        return self.fk(q) * self.grasp_point_offset
    def ik_map(self, T):
        return self.ik(T*self.grasp_point_offset.Inverse())[0]
    
    def _dvrk_measured_is_grasp_logic(self):
        js = self.get_signal("measured_jaw_js")
        if js<0 and js>np.deg2rad(-8.8):
            self.set_signal('measured_is_grasp', True)
        else:
            self.set_signal('measured_is_grasp', False)



class ECMClient(BaseClient):
    """
    ECM crtk topics client
    """
    def __init__(self, ros_node,
                       robot_type:str, 
                       is_left_cam=True, 
                       is_right_cam=False, 
                       is_left_point_cloud=False, 
                       is_right_point_cloud=False):

        super(ECMClient, self).__init__(ros_node)
        
        self.robot_type = robot_type

        # ros topics
        self.pubs['ecm_servo_jp'] = Publisher('/CRTK/ecm/servo_jp', JointState, queue_size=1) # we can only control joint position for ecm
        
        if self.robot_type == 'ambf':
            self.subs['camera_frame_state'] = Subscriber(self.topic_wrap('/CRTK/ecm/measured_cp'), PoseStamped, self._camera_frame_state_cb)
        elif self.robot_type == 'dvrk_2_0':
            self.set_signal('camera_frame_state', RPY2T(*[0,0,0, 0,0,0])) # by default, need to manually set if it is changed

           

        if is_left_cam:
            self.subs['cameraL_image'] = Subscriber(self.topic_wrap({"ambf": '/ambf/env/cameras/cameraL/ImageData',
                                                                "dvrk_2_0": '/camera/color/image_raw'}
                                                                [self.robot_type]
                                                                 ), numpy_msg(Image), self._cameraL_image_cb)

        if is_left_point_cloud:
            if robot_type == 'ambf':
                self.subs['cameraL_point_cloud'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraL/DepthData'), numpy_msg(PointCloud2), self._cameraL_image_depth_cb)
            elif robot_type == 'dvrk_2_0':
                self.subs['cameraL_image'] = Subscriber(self.topic_wrap('/camera/color/image_raw'), numpy_msg(Image), self._cameraL_image_cb)
                self.subs['cameraL_depth_image'] = Subscriber(self.topic_wrap('/camera/aligned_depth_to_color/image_raw'), numpy_msg(Image), self._cameraL_image_depth_cb2)
            else:
                raise Exception
                
        if is_right_cam:
            self.subs['cameraR_image'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraR/ImageData'), numpy_msg(Image), self._cameraR_image_cb)

        if is_right_point_cloud:
            self.subs['cameraR_point_cloud'] = Subscriber(self.topic_wrap('/ambf/env/cameras/cameraR/DepthData'), numpy_msg(PointCloud2), self._cameraR_image_depth_cb)

        

        # TODO: there is a minor for ecm js publisher
        # self.subs['measured_cam_js'] = Subscriber('/CRTK/ecm/measured_js', JointState, self._ecm_js_cb) # there is a minor for ecm js publisher
        self.ros_rate = Rate(5)  

    def move_ecm_jp(self, pos:List[float], time_out=30, threshold=0.001, ratio=1, count_break=3):
        """
        move camera joint position, with blocking (waiting)
        """
        msg = JointState()
        msg.position = pos
        self.pubs["ecm_servo_jp"].publish(msg)

        start = time.time()
        T_prv = self.get_signal('camera_frame_state')
        cnt = 0

        # block until the camera frame
        while True:
            self.ros_rate.sleep()
            T = self.get_signal('camera_frame_state')

            if T is None or T_prv is None:
                continue


            deltaT = T_prv.Inverse()*T
            theta, _ = deltaT.M.GetRotAngle()
            dp = deltaT.p.Norm()

            if time.time() - start > time_out:
                # print("move ecm, timeout")
                break
            
            # print(dp + theta)
            if dp + theta*ratio <threshold:
                cnt+=1
            else:
                cnt = 0
            
            if cnt >=count_break:
                # print("move ecm, stop")
                break

            T_prv = T

    def _cameraL_image_cb(self, data):
        """
        ros callback
        """
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.set_signal('cameraL_image',img)

    def _cameraL_image_depth_cb(self, data):
        """
        ros callback
        """
        # img = np.frombuffer(data.data, dtype=np.uint8)
        self.set_signal('cameraL_point_cloud', data)

    def _cameraL_image_depth_cb2(self, data):
        """
        ros callback
        """
        img = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width, -1)
        self.set_signal('cameraL_depth_image',img)
    
    def _cameraR_image_cb(self, data):
        """
        ros callback
        """
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.set_signal('cameraR_image',img)
    
    def _cameraR_image_depth_cb(self, data):
        """
        ros callback
        """
        # img = np.frombuffer(data.data, dtype=np.uint8)
        self.set_signal('cameraR_point_cloud', data)

    def _cameraL_local_state_cb(self, data):
        """
        ros callback
        """
        self.set_signal('cameraL_local_state',RigidBodyState2T(data))

    def _cameraR_local_state_cb(self, data):
        """
        ros callback
        """
        self.set_signal('cameraR_local_state',RigidBodyState2T(data))

    def _camera_frame_state_cb(self, data):
        """
        ros callback
        """
        self.set_signal('camera_frame_state', PoseStamped2T(data))



class SceneClient(BaseClient):
    """
    Scene crtk topics client
    """
    def __init__(self, ros_node):
        super(SceneClient, self).__init__(ros_node)
        # ros topics
        self.subs['measured_needle_cp'] = Subscriber(self.topic_wrap('/CRTK/Needle/measured_cp'), PoseStamped, self._needle_cp_cb)
        self.subs['measured_entry1_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry1/measured_cp'), PoseStamped, self._measured_entry1_cp_cb)
        self.subs['measured_entry2_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry2/measured_cp'), PoseStamped, self._measured_entry2_cp_cb)
        self.subs['measured_entry3_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry3/measured_cp'), PoseStamped, self._measured_entry3_cp_cb)
        self.subs['measured_entry4_cp'] = Subscriber(self.topic_wrap('/CRTK/Entry4/measured_cp'), PoseStamped, self._measured_entry4_cp_cb)

        self.subs['measured_exit1_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit1/measured_cp'), PoseStamped, self._measured_exit1_cp_cb)
        self.subs['measured_exit2_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit2/measured_cp'), PoseStamped, self._measured_exit2_cp_cb)
        self.subs['measured_exit3_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit3/measured_cp'), PoseStamped, self._measured_exit3_cp_cb)
        self.subs['measured_exit4_cp'] = Subscriber(self.topic_wrap('/CRTK/Exit4/measured_cp'), PoseStamped, self._measured_exit4_cp_cb) 

    def _needle_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_needle_cp', PoseStamped2T(data))

    def _measured_entry1_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry1_cp', PoseStamped2T(data))

    def _measured_entry2_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry2_cp', PoseStamped2T(data))
        
    def _measured_entry3_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry3_cp', PoseStamped2T(data))

    def _measured_entry4_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_entry4_cp', PoseStamped2T(data))


    #====
    def _measured_exit1_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit1_cp', PoseStamped2T(data))

    def _measured_exit2_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit2_cp', PoseStamped2T(data))
        
    def _measured_exit3_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit3_cp', PoseStamped2T(data))

    def _measured_exit4_cp_cb(self, data):
        """
        ros callback
        """
        self.set_signal('measured_exit4_cp', PoseStamped2T(data))



if __name__ == "__main__":
    # ros_node = init_node('ros_client_test',anonymous=True)
    # client = PSMClient(ros_node, 'psm1')
    # sleep(1)
    # print(client.sub_signals_names)
    # print(client.sub_signals['measured_base_cp'])



    # engine = ClientEngine()
    # engine.add_clients(['psm1', 'psm2'])
    # print(engine.client_names)
    # print(engine.get_signal('psm1', 'measured_base_cp'))
    # print(engine.get_signal('psm2', 'measured_base_cp'))

    # print("============ move ecm ")
    # engine.add_clients(['ecm'])
    # engine.clients['ecm'].servo_cam_jp([0,0,-0.9,0])

    # print(engine.get_signal('ecm','cameraL_image').shape)





    engine = ClientEngine()
    engine.add_clients(['psm2'])
    engine.start()

    q0 = [-0.5656515955924988, -0.15630173683166504, 1.3160043954849243, -2.2147457599639893, 0.8174221515655518, -1]

    sleep_time = 0.3
    engine.clients['psm2'].servo_jp(q0, interpolate_num=100)
    engine.clients['psm2'].close_jaw()
    engine.clients['psm2'].sleep(sleep_time)

    T_g_w_msr = engine.clients['psm2'].T_g_w_msr
    deltaT = RPY2T(*[0.2,0,0,0,0,0])
    engine.clients['psm2'].servo_tool_cp(deltaT * T_g_w_msr, interpolate_num=100)
    engine.clients['psm2'].open_jaw()
    engine.clients['psm2'].sleep(sleep_time)
    
    T_g_w_msr = engine.clients['psm2'].T_g_w_msr
    deltaT = RPY2T(*[0,0.2,0,0,0,0])
    engine.clients['psm2'].servo_tool_cp(deltaT * T_g_w_msr, interpolate_num=100)
    engine.clients['psm2'].close_jaw()
    engine.clients['psm2'].sleep(sleep_time)

    T_g_w_msr = engine.clients['psm2'].T_g_w_msr
    deltaT = RPY2T(*[0,0,0.1,0,0,0])
    engine.clients['psm2'].servo_tool_cp(deltaT * T_g_w_msr, interpolate_num=100)
    engine.clients['psm2'].open_jaw()
    engine.clients['psm2'].sleep(sleep_time)


    print("ctrl+c to stop")
    spin()
    engine.close()