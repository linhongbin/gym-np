from gym_suture.env.suture_env import SurgialChallengeEnv
from gym_suture.param import tune_params
from gym_suture.tool.segment import SegmentEngine
from gym_suture.tool.common import resize_img
import numpy as np
import cv2
import time
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render', type=str, default='origin_rgb') # [origin_rgb, origin_depth, image_preprocess, seg_gripper, seg_needle]
parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--platform', type=str, default='phantom') #[cuboid, phantom]
parser.add_argument('--preprocess-type', type=str, default="segment_script") # ['segment_script','segment_net']
parser.add_argument('--image-type', type=str, default="zoom_needle_boximage") #[zoom_needle_gripper_boximage, zoom_needle_boximage]
parser.add_argument('--arm', type=str, default='psm2')  
parser.add_argument('--seg-obj', type=str, default='None') #["none", "gripper", "needle", "all"] 
parser.add_argument('--segment-net-file', type=str, default="none") 

parser.add_argument('--hz', type=int, default=5) # [rgb, depth]
parser.add_argument('--savedir', type=str, default="./data/render_obs_test")
parser.add_argument('--resize', action='store_true')
parser.add_argument('--beta', type=float, default=0.3)


args = parser.parse_args()
param_name = args.robot + '_' + args.platform
env = SurgialChallengeEnv(
                action_arm_device = args.arm,
                **(tune_params[param_name].suture_env)
                )
seg_engine = SegmentEngine(robot_type=args.robot,
                           process_type=args.preprocess_type,
                           image_type=args.image_type,
                           segment_net_file = None if args.segment_net_file=="none" else args.segment_net_file,
                           )
print('initializing...')
time.sleep(3)
print("press q to exit")
path = Path(args.savedir)
while True:
    obs = env._get_obs()

    if args.render == 'origin_rgb':
        render_frame = obs['image']
    elif args.render == 'origin_depth':
        render_frame = np.concatenate([obs['depth_xyz']]*3, axis=2)
    elif args.render == 'seg_gripper' or args.render == 'seg_needle':
        render_frame = seg_engine.render_mask(obs['image'], render_method={"seg_gripper": "gray_gripper",
                                                                    "seg_needle": "gray_needle",}[args.obs])
    elif args.render == 'image_preprocess':
        results = seg_engine.process_image(obs['image'], depth=obs['depth_xyz'])
        render_frame = results['image']
    else:
        raise NotImplementedError
    

    if args.seg_obj in ["gripper", "needle", "all"]:
        seg_render_image = seg_engine.segment_image(segment_input_im=obs["image"], render_im=render_frame, segment_object=args.seg_obj)
        render_frame = cv2.addWeighted(seg_render_image, 1, render_frame, args.beta, 0.0)
        
    if args.resize:
        render_frame = cv2.resize(render_frame, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)


    rende_frame_resize = cv2.resize(render_frame, (800, 800), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('preview', cv2.cvtColor(rende_frame_resize, cv2.COLOR_RGB2BGR)) # Display the resulting frame
    cv2.setWindowTitle('preview', 'press q to exit, press other keys to step')
    k = cv2.waitKey(int(1/args.hz*1000))
    if k& 0xFF == ord('q'):    # Esc key to stop
        break
    elif k& 0xFF == ord('s'):
        path.mkdir(exist_ok=True, parents=True)
        file = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(path / file) + '.png', cv2.cvtColor(render_frame, cv2.COLOR_RGB2BGR)) 
        m,n,r = raw.shape
        out_arr = np.column_stack((np.repeat(np.arange(m),n),raw.reshape(m*n,-1)))
        out_df = pd.DataFrame(out_arr)
        out_df.to_csv(str(path / file) + '.csv')
    else:
        # print("")
        pass

env.close()