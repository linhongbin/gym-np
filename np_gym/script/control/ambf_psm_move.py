from gym_np.tool.ros_client import PSMClient
from gym_np.tool.common import RPY2T, T2RPY, Quaternion2T
import rospy
from time import sleep
from numpy import pi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-x', type=float, default=0)
parser.add_argument('-y', type=float, default=0.1)
parser.add_argument('-z', type=float, default=0.7)
args = parser.parse_args()


ros_node = rospy.init_node('ros_client_engine',anonymous=True)
client = PSMClient(ros_node,
                   default_servo_type='servo_jp',
                   arm_name='psm2',
                    robot_type='ambf', 
                    # default_kin_engine='peter_corke',
                   default_kin_engine='surgical_challenge',
                   qlim_MarginRatio=0.03,
                    jaw_qlim_MarginRatio=0.03,
                    ignore_qlim=True,
                   )


client.start()
# print("reseting pose...")
# client.reset_pose()
# sleep(3)
pose = [args.x, args.y, args.z, pi,0,0]
T = RPY2T(*pose)
client.servo_tool_cp(T, interpolate_num=100, clear_queue=False)
client.wait()

# print('finish, type ctrl +c to stop')
# rospy.spin()
client.close()
