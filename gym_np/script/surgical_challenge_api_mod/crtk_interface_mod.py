from gym_np.script.surgical_challenge_api_mod.launch_crtk_interface import Client, PSMCRTKWrapper, ECMCRTKWrapper, SceneCRTKWrapper, SceneManager, get_boolean_from_opt, Options
# from surgical_robotics_challenge.psm_arm import PSM
from gym_np.script.surgical_challenge_api_mod.psm_arm_mod import PSM_Mod
import time
import rospy
from std_msgs.msg import Bool
from argparse import ArgumentParser
from geometry_msgs.msg import TransformStamped
from gym_np.tool.common import TransformStamped2T
from sensor_msgs.msg import JointState

class PSMCRTKWrapperModified(PSMCRTKWrapper):
    def __init__(self, client, name, namespace, add_joint_errors=False):
        super(PSMCRTKWrapperModified, self).__init__(client, name, namespace)
        self.arm = PSM_Mod(client, name, add_joint_errors=add_joint_errors)
        self.measured_jaw_js_pub = rospy.Publisher(namespace + '/' + name + '/' + 'measured_jaw_js', JointState,
                                               queue_size=1)
            
        self.measured_grasp_pub = rospy.Publisher(namespace + '/' + name + '/' + 'is_grasp', Bool,
                                               queue_size=1)
        # self.servo_marker_cp_sub = rospy.Subscriber(namespace + '/' + name + '/' + 'servo_marker_cp', TransformStamped,
        #                                      self.servo_marker_cp, queue_size=1)
    
    # def servo_jp_cb(self, js): # there is bug in original one
    #     self.arm.servo_jp(list(js.position))

    # def servo_marker_cp(self, cp):
    #     frame = TransformStamped2T(cp)
    #     # print(frame)
    #     if self.arm.target_IK is not None:
    #         # print("set")
    #         self.arm.target_IK.set_pos(frame.p[0], frame.p[1], frame.p[2])
    #         self.arm.target_IK.set_rpy(frame.M.GetRPY()[0], frame.M.GetRPY()[1], frame.M.GetRPY()[2])
    def publish_gripper(self):
        msg = Bool()
        msg.data = self.arm.grasped[0]
        self.measured_grasp_pub.publish(msg)
        
    def publish_jaw_js(self):
        msg =JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = self.arm.measured_jaw_js()
        self.measured_jaw_js_pub.publish(msg)
        
    def run(self):
        self.publish_js()
        self.publish_cs()
        self.publish_jaw_js()
        self.publish_gripper()


class SceneManagerModified(SceneManager):
    def __init__(self, options):
        self.client = Client("ambf_surgical_sim_crtk_node")
        self.client.connect()
        time.sleep(0.2)
        self._components = []
        if options.run_psm_one is True:
            print("Launching CRTK-ROS Interface for PSM1 ")
            psm1 = PSMCRTKWrapperModified(self.client, 'psm1', options.namespace)
            self._components.append(psm1)
        if options.run_psm_two is True:
            print("Launching CRTK-ROS Interface for PSM2 ")
            psm2 = PSMCRTKWrapperModified(self.client, 'psm2', options.namespace)
            self._components.append(psm2)
        if options.run_psm_three is True:
            print("Launching CRTK-ROS Interface for PSM3 ")
            psm3 = PSMCRTKWrapperModified(self.client, 'psm3', options.namespace)
            self._components.append(psm3)
        if options.run_ecm:
            print("Launching CRTK-ROS Interface for ECM ")
            ecm = ECMCRTKWrapper(self.client, 'ecm', options.namespace)
            self._components.append(ecm)
        if options.run_scene:
            print("Launching CRTK-ROS Interface for Scene ")
            scene = SceneCRTKWrapper(self.client, options.namespace)
            self._components.append(scene)

        self._rate = rospy.Rate(options.rate)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--one', action='store', dest='run_psm_one', help='RUN PSM1', default=True)
    parser.add_argument('--two', action='store', dest='run_psm_two', help='RUN PSM2', default=True)
    parser.add_argument('--three', action='store', dest='run_psm_three', help='RUN PSM3', default=False)
    parser.add_argument('--ecm', action='store', dest='run_ecm', help='RUN ECM', default=True)
    parser.add_argument('--scene', action='store', dest='run_scene', help='RUN Scene', default=True)
    parser.add_argument('--ns', action='store', dest='namespace', help='Namespace', default='/CRTK')
    parser.add_argument('--rate', action='store', dest='rate', help='Rate of Publishing', default=120)

    parsed_args = parser.parse_args()
    print('Specified Arguments')
    print(parsed_args)
    options = Options()

    options.run_psm_one = get_boolean_from_opt(parsed_args.run_psm_one)
    options.run_psm_two = get_boolean_from_opt(parsed_args.run_psm_two)
    options.run_psm_three = get_boolean_from_opt(parsed_args.run_psm_three)
    options.run_ecm = get_boolean_from_opt(parsed_args.run_ecm)
    options.run_scene = get_boolean_from_opt(parsed_args.run_scene)

    options.namespace = parsed_args.namespace
    options.rate = parsed_args.rate

    sceneManager = SceneManagerModified(options)
    sceneManager.run()




