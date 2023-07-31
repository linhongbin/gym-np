import rospy
import numpy as np  
from ds4_driver.msg import Status, Feedback
import time
class DS_Controller():
    def __init__(self, sig_keys=None,
                      exit_thres = 0.3, # reset position signals will reset to zero, we mearsure max(abs(value)) of signals < exit_thres if reset
                      enter_thres = 0.7, # signal will set to 1 or -1 when a button or stick is pressed, we mearsure max(abs(value)) of signals > enter_thres if press
                      ):
        try: # avoid multiple call
            self._node = rospy.init_node('DS_Controller', anonymous=True)
        except:
            print("skip init node")
            
        topic_name = '/status'
        topics = [topic[0] for topic in rospy.get_published_topics()]
        if not (topic_name in topics):
            raise Exception("topic {} does not exist, please publisher is running".format(topic_name))
        self._sub = rospy.Subscriber("/status", Status, self.cb)
        self._pub = rospy.Publisher("/set_feedback", Feedback, queue_size=1)
        

        
        self.data = {}
        self.sig_keys = sig_keys or {
            "button_dpad_left": 1,
            "button_dpad_right": 1,
            "button_dpad_down": 1,
            "button_dpad_up": 1,
            "button_cross": 1,
            "button_triangle": 1,
            "button_circle": 1,
            "button_square": 1,
            "button_r2": 1,
            "button_l2": 1,
            "axis_left_x":2, # dual direction
        }
        self.exit_thres = exit_thres
        self.enter_thres = enter_thres
        print("init ds contorller...")
        time.sleep(2)
        print("finish")

    def cb(self, data):
        for k in self.sig_keys.keys():
            self.data[k] = getattr(data, k)
            # self.data['axis_left_x'] = data.axis_left_x
            # self.data['axis_left_y'] = data.axis_left_y
            # self.data['axis_right_x'] = data.axis_right_x
            # self.data['axis_right_y'] = data.axis_right_y
            
    def get_discrete_cmd(self, n_max=None):
        sig_keys = list(self.sig_keys.keys())
        if n_max is not None:
            _n = 0
            is_trunc = False
            for i, k in enumerate(sig_keys):
                _n += self.sig_keys[k]
                if _n > n_max:
                    # print("stop ", _n, i)
                    is_trunc = True
                    break
            if is_trunc:
                sig_keys = sig_keys[:i]  
        act = 0
        is_reset = True
        while is_reset:
            sig = np.array([self.data[k] for k in sig_keys])
            if np.max(np.abs(sig)) > self.enter_thres:
                is_reset = False
                idx = np.argmax(np.abs(sig))
                act = 0
                for i in range(idx):
                    act += self.sig_keys[sig_keys[i]]
                sgn = sig < 0
                act += int(sgn[idx])
        
        self.rumble_on(big_rum=0.0,small_rum=0.5)
        while not is_reset:
            sig = np.array([self.data[k] for k in sig_keys])
            if np.max(np.abs(sig)) < self.exit_thres:
                is_reset = True
        self.rumble_off()
        # print(act)
        return act
    
    def rumble_on(self, big_rum=0.5, small_rum=0.5):
        self._assert_0_1(big_rum)
        self._assert_0_1(small_rum)
        msg = Feedback()
        msg.set_rumble = True
        msg.rumble_small =big_rum
        msg.rumble_big =small_rum
        self._pub.publish(msg)

    def rumble_off(self):
        msg = Feedback()
        msg.set_rumble = False
        self._pub.publish(msg)

    def led_on(self, r=0.1,g=0.1,b=0.1):
        self._assert_0_1(r)
        self._assert_0_1(g)
        self._assert_0_1(b)
        msg = Feedback()
        msg.set_led = True
        msg.led_r = float(r)
        msg.led_g = float(g)
        msg.led_b = float(b)
        self._pub.publish(msg)
    def led_off(self):
        self.led_on(r=0.0,g=0.0,b=0.0)
    
    def _assert_0_1(self,x):
        assert x >=0 and x <=1

        






if __name__ == '__main__':
    con = DS_Controller()
    r = rospy.Rate(1) # 10hz
    rospy.sleep(1)
    while not rospy.is_shutdown():
        # print(con.data['button_r2'])
        action = con.get_discrete_act(9)
        print(action)
        r.sleep()

