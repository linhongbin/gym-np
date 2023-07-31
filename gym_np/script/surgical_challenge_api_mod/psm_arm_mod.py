from surgical_robotics_challenge.psm_arm import PSM



class PSM_Mod(PSM):
    def __init__(self, client, name, add_joint_errors=True, save_jp=False):
        super().__init__(client, name, add_joint_errors, save_jp)
    
    def measured_jaw_js(self):
        # j6 = self.base.get_joint_pos(6)
        j6= self.base.get_joint_pos('toolyawlink-toolgripper1link')
        q = [j6]
        return q