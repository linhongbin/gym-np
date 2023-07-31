# import rospy
# from sensor_msgs.msg import JointState
# from queue import Queue


# js_dsr = None
# interpolate_num = 
# def move_jp_cb(data):
#     js_dsr = data.position
#     np.linspace(self.jaw_dsr, jaw_jp_dsr, interpolate_num).tolist()
# if __name__ == "__main__":
#     jp_move_sub = rospy.Subscriber("/CRTK/psm2/move_jp", JointState,
#                                                 move_jp_cb, queue_size=1)

#     jp_servo_pub = rospy.Publisher("/CRTK/psm2/servo_jp", JointState,queue_size=1)
    
#     rate = rospy.Rate(100)
    
#     while not rospy.is_shutdown():
#         rate.sleep()