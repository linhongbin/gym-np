import argparse
from gym_suture.env.wrapper import make_env
from time import sleep
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('-s', type=int, default=32)
parser.add_argument('-r', type=int, default=1) # [random, zero, oracle]
parser.add_argument('-t', type=int, default=50) # [random, zero, oracle]

parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--platform', type=str, default='phantom') #[cuboid, phantom]
parser.add_argument('--action', type=str, default='random') # [random, zero, oracle]
parser.add_argument('--arm', type=str, default='psm2') # [random, zero, oracle]
parser.add_argument('--idle-action', action='store_true')
parser.add_argument('--savedir', type=str, default='./data/segmentation/ambf/sanity_test_failed') 


args = parser.parse_args()

is_ds4_oracle = (args.robot=='dvrk') and (args.action=='oracle')
savedir = Path(args.savedir) / time.strftime("%Y%m%d-%H%M%S")
savedir.mkdir(parents=True,exist_ok=True)

env = make_env(
                robot_type=args.robot,
             platform_type=args.platform, #[cuboid, phantom]
                    is_visualizer=False, 
                        is_idle_action=args.idle_action, 
                        preprocess_type="segment_net",
                        is_ds4_oracle=is_ds4_oracle,
                        action_arm_device=args.arm,
                        obs_type ="image",
                        timelimit=args.t)
env.seed = args.s

cnt = 0
for i in tqdm(range(args.r)):
    env.reset() 
    sleep(2)
    done =False
    while not done:
        # action = 0
        # action = env.get_oracle_action(noise_scale=0.15)
        # action = env.get_oracle_action(noise_scale=0.0)
        # action = env.action_space.sample()
        if args.action == 'random': 
            action = env.action_space.sample()
        elif args.action == 'zero': 
            action = 0
        elif args.action == 'oracle':
            action = env.get_oracle_action() 
        obs, reward, done, info = env.step(action)
    
    if obs['state'] == 5:
        cnt +=1
        done = False 
        print(f"get {cnt} bad images..")
        file = time.strftime("%Y%m%d-%H%M%S") + ".png"
        cv2.imwrite(str(savedir / file), cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR))
env.close()

