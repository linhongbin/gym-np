from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH # Modifed DH
from spatialmath import SE3
from numpy import pi
import numpy as np
import time
from gym_np.tool.common import RPY2T, T_2_SE3
PI_2 = pi/2

class PSM_KIN():
    def __init__(self, robot_type) -> None:
        pass

        self.num_links = 7
        
        if robot_type == 'ambf':
            self.L_rcc = 4.389  # From dVRK documentation x 10
            self.L_tool = 4.16  # From dVRK documentation x 10
            self.L_pitch2yaw = 0.09  # Fixed length from the palm joint to the pinch joint
            self.L_yaw2ctrlpnt = 0.106  # Fixed length from the pinch joint to the pinch tip
            # Delta between tool tip and the Remote Center of Motion
            self.L_tool2rcm_offset = 0.229
            self.qmin = np.deg2rad([-91.96, -60, 0.0, -175,-80, -80]) # the last 3 related to tool limits, we use large needle driver 400006
            self.qmax = np.deg2rad([ 91.96,  60, 0.0,  175, 80,  80])
            self.qmin[2] = 0
            self.qmax[2] = 2.4 # 10x scale up
            
            self.jaw_qmin = np.deg2rad(-20)
            self.jaw_qmax = np.deg2rad(80)

        elif robot_type == 'dvrk_2_0': 
            self.L_rcc = 0.4389  # refer to large needle driver 4006
            self.L_tool = 0.4162  
            self.L_pitch2yaw = 0.0091 
            self.L_yaw2ctrlpnt = 0.0106  
            self.L_tool2rcm_offset = 0.0229
            self.qmin = np.deg2rad([-270, -53, 0.0, -260,-80, -80])
            self.qmax = np.deg2rad([ 270,  53, 0.0,  260, 80,  80])
            self.qmin[2] = 0
            self.qmax[2] = 0.24 # 10x scale up
            
            self.jaw_qmin = np.deg2rad(-20)
            self.jaw_qmax = np.deg2rad(80)
            


        # joint limits
        # refer to dVRK PSM + large needle 4006


        self.tool_T = np.array([[0, -1,  0, 0],
                                [0,   0,  1, self.L_yaw2ctrlpnt],
                                [-1,  0,  0, 0],
                                [0,  0,  0, 1]])
        
        

        self.build_kin()



    def build_kin(self):
        self.robot = DHRobot(
            [
                RevoluteMDH(alpha=PI_2, a=0,               d=0,                 offset=PI_2,   qlim=np.array([self.qmin[0], self.qmax[0] ])),
                RevoluteMDH(alpha=-PI_2,a=0,               d=0,                 offset=-PI_2,   qlim=np.array([self.qmin[1], self.qmax[1] ])),
                PrismaticMDH(alpha=PI_2,a=0,               theta=0,             offset=-self.L_rcc,   qlim=np.array([self.qmin[2], self.qmax[2]])),
                RevoluteMDH(alpha=0,    a=0,               d=self.L_tool,       offset=0,       qlim=np.array([self.qmin[3], self.qmax[3]])),
                RevoluteMDH(alpha=-PI_2,a=0,               d=0,                 offset=-PI_2,   qlim=np.array([self.qmin[4], self.qmax[4]])),
                RevoluteMDH(alpha=-PI_2,a=self.L_pitch2yaw,d=0,                 offset=-PI_2,   qlim=np.array([self.qmin[5], self.qmax[5]]))
            ], name="PSM")
        
        self.robot.tool = SE3(self.tool_T)
        
        # self.robot.qlim = np.array([self.qmin, self.qmax])
        
        self._qlim = (self.robot.qlim[0], self.robot.qlim[1])
    def fk(self, qs):
        assert len(qs) == 6
        return self.robot.fkine(qs)

    def ik(self,T_dsr, q0, method="LM"):
        assert len(q0) == 6
        if method == "LM":
            result = self.robot.ikine_LM(T=T_dsr, q0=q0)
            return result.q.tolist(), result.success

        elif method == "JNT_LMIT":
            result = self.robot.ikine_min(T=T_dsr, q0=q0, qlim=True)
            return result.q.tolist(), result.success
        else:
            raise NotImplementedError


    def jacob(self, qs):
        assert len(qs) == 6
        _qs = qs + [0]
        return self.robot.jacob0(_qs)

    def sample_q(self):
        """ sample joint position within joint limits"""
        return np.random.uniform(low=self.qlim[0], high=self.qlim[1], size=(6,)).tolist()
    
    @property
    def qlim(self):
        return self._qlim[0], self._qlim[1]

    def is_out_qlim(self, q, q_min_margin=None, q_max_margin=None, margin_ratio=None):
        _q = np.array(q)
        if (q_min_margin is None) and (q_max_margin is None) and (margin_ratio is None):
            _q_min_margin =  np.zeros(len(q))
            _q_max_margin =  np.zeros(len(q))
        elif margin_ratio is not None:
            _q_min_margin = (self.qlim[1] - self.qlim[0])*margin_ratio/2
            _q_max_margin = (self.qlim[1] - self.qlim[0])*margin_ratio/2
        else:
            _q_min_margin = np.array(q_min_margin)
            _q_max_margin = np.array(q_max_margin)

        result = np.logical_or(_q<self.qlim[0]+_q_min_margin, _q>self.qlim[1]-_q_max_margin)
        return np.sum(result)!=0, result
    
    def is_out_jaw_qlim(self, q, margin_ratio):
        _q = q
        _q_min_margin = (self.jaw_qmax - self.jaw_qmin)*margin_ratio/2
        _q_max_margin = (self.jaw_qmax - self.jaw_qmin)*margin_ratio/2
        
        result = np.logical_or(_q<self.jaw_qmin+_q_min_margin, _q>self.jaw_qmax-_q_max_margin)
        return result
    
