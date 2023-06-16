from gym_suture.tool.ros_client import PSMClient
from gym_suture.tool.common import RPY2T, T2RPY, Quaternion2T
import rospy
from time import sleep
from numpy import pi

ros_node = rospy.init_node('ros_client_engine',anonymous=True)
client = PSMClient(ros_node,
                   default_servo_type='servo_jp',
                   arm_name='psm2',
                    robot_type='ambf', 
                    # default_kin_engine='peter_corke',
                   default_kin_engine='surgical_challenge',
                   qlim_MarginRatio=0.03,
                    jaw_qlim_MarginRatio=0.03
                   )


client.start()
print("reseting pose...")
client.reset_pose(interpolate_num=50)


# origin_pos, _ = T2RPY(client.T_g_w_dsr)

# T_origin = RPY2T(*origin_pos.tolist(),*[pi, 0, pi/2])
# client.servo_tool_cp(T_origin, 200)
# client.wait()

# client.open_jaw()
# client.wait()
# client.close_jaw()
# client.wait()

# # deltaT =  RPY2T(*[0,0.05,0,0,0,0])
# # client.servo_tool_cp(T_origin *deltaT, 300)
# # client.wait()

# T =  RPY2T(*[0.1,0,0, 0,0, 0])
# client.servo_tool_cp(T*T_origin, 200)
# client.wait()

# client.servo_tool_cp(T_origin, 200)
# client.wait()



sleep(3)

# print('finish, type ctrl +c to stop')
# rospy.spin()
client.close()
