from gym_suture.tool.ros_client import PSMClient
from gym_suture.tool.common import RPY2T, T2RPY, Quaternion2T
import rospy
from time import sleep
from numpy import pi
import argparse
from gym_suture.tool.input import DS_Controller
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument('--obs', type=str, default='rgb') # [rgb, depth]
parser.add_argument('--arm', type=str, default='psm1') 

parser.add_argument('--origin-x-ratio', type=float, default=0.5) 
parser.add_argument('--origin-y-ratio', type=float, default=0.5) 
parser.add_argument('--origin-z-ratio', type=float, default=0.65)
parser.add_argument('--gripper-initmargin', type=float, default=0.01)
parser.add_argument('--needle-initmargin', type=float, default=0.02)
parser.add_argument('--needle-z-initratio', type=float, default=0.3) 
args = parser.parse_args()




ros_node = rospy.init_node('ros_client_engine',anonymous=True)
client = PSMClient(ros_node,
                    arm_name=args.arm,
                    default_servo_type='servo_jp',
                robot_type='dvrk_2_0',
                default_kin_engine='surgical_challenge',
                is_measure_is_grasp=False,
                qlim_MarginRatio=0.01,
                jaw_qlim_MarginRatio=0.01,
                manual_set_base_rpy=[0,0,0, 0,0, 0],
                   )
print("waiting for ds input, press any key to forward.....")
in_device = DS_Controller()

client.start()
# print("reseting pose...")
# client.reset_pose()
rospy.sleep(2)
# while not rospy.is_shutdown():
def get_pose():
    cmd = in_device.get_discrete_cmd()
    # print("pos:",client.T_g_w_msr.p)
    return client.T_g_w_msr

Ts = {}
print(f"Drag {args.arm} to following position in floating mode....")
print("move to left limit")
in_device.led_on(r=1)
v1 = get_pose()
print("move to right limit")
v2 = get_pose()
Ts['ws_lim_lr'] =sorted([v1.p.x(), v2.p.x()])

in_device.led_on(g=1)
print("move to forward limit")
v1 = get_pose()
print("move to backward limit")
v2 = get_pose()
Ts['ws_lim_fb'] =sorted([v1.p.y(), v2.p.y()])


in_device.led_on(b=1)
print("move to upward limit")
v1 = get_pose()
print("move to downward limit")
v2 = get_pose()
Ts['ws_lim_ud'] =sorted([v1.p.z(), v2.p.z()])


in_device.led_on(g=0.8,b=0.8)
print("move to upward right limit")
v1 = get_pose()
print("move to downward right limit")
v2 = get_pose()
delta = v1.p -v2.p
if np.abs(delta.y())>1e-16:
    theta = np.arctan(delta.x()/delta.y()) 
else:
    theta = np.arctan(delta.x()*np.sign(delta.y()) * 1e10) 
print(f"delta theta is {theta} rad, {np.rad2deg(theta)} degree")
# in_device.led_on(b=1)
in_device.led_off()





ratio_func = lambda x,y, rate: x*(1-rate)+y*rate
print("=======get results==========")
print("\"ws_x_low\":{:.3f},".format(Ts['ws_lim_lr'][0]))
print("\"ws_x_high\":{:.3f},".format(Ts['ws_lim_lr'][1]))
print("\"ws_y_low\":{:.3f},".format(Ts['ws_lim_fb'][0]))
print("\"ws_y_high\":{:.3f},".format(Ts['ws_lim_fb'][1]))
print("\"ws_z_low\":{:.6f},".format(Ts['ws_lim_ud'][0]))
print("\"ws_z_high\":{:.3f},".format(Ts['ws_lim_ud'][1]))
init_origin_pos = [ratio_func(Ts['ws_lim_lr'][0],Ts['ws_lim_lr'][1], args.origin_x_ratio),
                   ratio_func(Ts['ws_lim_fb'][0],Ts['ws_lim_fb'][1], args.origin_y_ratio),
                   ratio_func(Ts['ws_lim_ud'][0],Ts['ws_lim_ud'][1], args.origin_z_ratio),]

# print()
# print("\"init_orgin_RPY\":[{:.2f},{:.2f}, {:.2f}, np.pi, 0, np.pi/2],".format(*init_origin_pos))

print()
print(f"\"manual_set_base_rpy\":[0,0,0, 0,0,{theta}],")

init_origin_pose = init_origin_pos
init_origin_pose.extend([np.pi, 0, np.pi/2])


qs_reset = client.ik_map(RPY2T(*init_origin_pose))
print()
print("\"q_dsr_reset\":[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},],".format(*qs_reset))


print()
print("\"init_pos_bound_dict\":{{ 'x':[{:.3f}, {:.3f}],'y':[{:.3f}, {:.3f}],'z':[{:.3f}, {:.3f}]}},".format(\
Ts['ws_lim_lr'][0]+args.gripper_initmargin,
Ts['ws_lim_lr'][1]-args.gripper_initmargin,
Ts['ws_lim_fb'][0]+args.gripper_initmargin,
Ts['ws_lim_fb'][1]-args.gripper_initmargin,
init_origin_pos[2],
init_origin_pos[2],
))

print()
init_z_needle = ratio_func(Ts['ws_lim_ud'][0],Ts['ws_lim_ud'][1],args.needle_z_initratio)
print("\"needle_init_pose_bound\":{{ 'low':[{:.3f}, {:.3f},{:.3f}, 0,0, -np.pi,],\n              'high':[{:.3f}, {:.3f},{:.3f}, 0,0., np.pi,],}},".format(\
Ts['ws_lim_lr'][0]+args.needle_initmargin,
Ts['ws_lim_fb'][0]+args.needle_initmargin,
init_z_needle,
Ts['ws_lim_lr'][1]-args.needle_initmargin,
Ts['ws_lim_fb'][1]-args.needle_initmargin,
init_z_needle
))
print("==================================")

# print(Ts['ws_lim_lr'])
# Ts['ws_lim_fb'] = [v1.p.y(), v1.p.y()]
# Ts['ws_lim_ud'] = [v1.p.z(), v1.p.z()]

# rospy.spin()
client.close()