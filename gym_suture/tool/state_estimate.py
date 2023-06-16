from enum import Enum
import numpy as np
import operator
class TaskStates(Enum):
    """ States labeled"""
    InProcess = 0.0 
    EndSucess = 1.0
    EndFail_TimeLimit = 2.0
    EndFail_Segment = 3.0
    EndFail_WorkSpace = 4.0 # value is less, priority is higher
    EndFail_JointLimit = 5.0



class StateEstimator():
    def __init__(self, robot_type, task_type, 
                    states_type='normal', 
                    method="tracking",
                    is_force_inprogress=False,
                    method_args={}, 
                    verbose=True) -> None:
        self.robot_type = robot_type
        self.task_type = task_type
        # TaskStates = {"normal":States}[states_type]
        self.method = method
        self.method_args = method_args
        self.verbose = verbose
        self.is_force_inprogress =is_force_inprogress
        
    def reset(self):
        self.prev_sig = None
        self.sucess_count = 0
        
    def __call__(self, obs, info, action):
        if len(info['done']) > 0:
            done_reason = [list(i.keys())[0] for i in info['done']]
            dict_map = {
                'q_out':TaskStates.EndFail_JointLimit,
                'ws_out':TaskStates.EndFail_WorkSpace,
                'timelimit_out':TaskStates.EndFail_TimeLimit,
                'segment_fail':TaskStates.EndFail_Segment,
            }
            done_dict = {dict_map[k]: dict_map[k].value for k in done_reason}
            done_ = sorted(done_dict.items(),key=operator.itemgetter(1),reverse=False) 
            return done_[0][0]
        elif self.is_force_inprogress:
            return TaskStates.InProcess
        else:
            if self.task_type == 'needle_picking':
                if self.method =="tracking":
                    is_sucess = ((info['needle_pos'] - info['needle_initial_pos'])[2]>info['success_lift_height'])\
                                                and info['is_grasp']
                elif self.method =="sensor":
                    if action==5 and info['is_grasp']:
                        self.sucess_count +=1
                    else:
                        self.sucess_count = 0
                    is_sucess = self.sucess_count >=3
                elif self.method =="segment_stat":
                    metric = np.array([info['needle_x_mean'] - info['gripper_x_mean'],
                                                    info['needle_y_mean'] - info['gripper_y_mean'],
                                                    info['needle_value_mean'] - info['gripper_value_mean'],
                                                    ])
                    # dis = np.linalg.norm(metric)
                    metric_weight = np.array(self.method_args["metric_weights"])
                    metric = np.abs(metric)* metric_weight
                    is_sucess = False
                    if self.prev_sig is not None: 
                        delta = metric-self.prev_sig
                        # print(f"metric signal {metric}")

                        # print(np.linalg.norm(delta))
                        # print(self.method_args["sucess_box_dist_thres"])
                        if action==5:
                            if self.verbose:
                                print("**********************")
                                print(f"metric: {metric} | metric delta: {delta} | delta norm {np.linalg.norm(delta)}")
                                print("**********************")
                            if np.linalg.norm(delta) < self.method_args["sucess_box_dist_thres"]  and info["is_clutch_engage"]:
                                self.sucess_count += 1
                                if self.sucess_count >= self.method_args["sucess_sig_count"]:
                                    is_sucess=True 
                        else:
                            self.sucess_count = 0               
                    self.prev_sig = metric

                else:
                    raise NotImplementedError
                if is_sucess:
                    return TaskStates.EndSucess
                else:              
                    return TaskStates.InProcess

            else:
                raise NotImplementedError
    
    
if __name__ == '__main__':
    print(States.InProcess in list(States))
    print(tuple(States))
