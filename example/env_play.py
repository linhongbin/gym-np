import argparse
from gym_np.env.wrapper import GymSutureEnv
from time import sleep
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=int, default=32)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--platform', type=str, default='phantom') #[cuboid, phantom]
parser.add_argument('--needle', type=str, default='standard') #
parser.add_argument('--action', type=str, default='oracle') # [random, zero, oracle]
parser.add_argument('--arm', type=str, default='psm2') # [psm1, psm2]
parser.add_argument('--preprocess-type', type=str, default='segment_script') # [segment_net, mixdepth,origin, segment_script]
parser.add_argument('--image-type', type=str, default='zoom_needle_boximage') #[zoom_needle_gripper_boximage, zoom_needle_boximage]
parser.add_argument('--clutch', type=int, default=6)
parser.add_argument('--timelimit', type=int, default=130)
parser.add_argument('--render-mode', type=str, default="human") # [human_depth, human]
parser.add_argument('--segment-net-file', type=str, default="none") 
parser.add_argument('--reset', type=str, default="manual")
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--noise-depth', type=float, default=0)

parser.add_argument('--no-visual', action='store_true')
parser.add_argument('--ds4', action='store_true')
parser.add_argument('--idle-action', action='store_true')
parser.add_argument('--resize', action='store_true')



args = parser.parse_args()

is_ds4_oracle = (args.robot=='dvrk') and (args.action=='oracle')
is_ds4_oracle = is_ds4_oracle or args.ds4

render_mode = args.render_mode
env = GymSutureEnv(
            robot_type=args.robot,
             platform_type=args.platform, #[cuboid, phantom]
             needle_type=args.needle,
             preprocess_type=args.preprocess_type, 
             image_type=args.image_type,
            #  scalar2image_obs_key=["gripper_state", "state"],
             action_arm_device=args.arm,
            reset_needle_mode=args.reset,
             clutch_start_engaged=args.clutch,
             resize_resolution=64 if args.resize else -1,
            timelimit=args.timelimit, 
            segment_net_file=None if args.segment_net_file=="none" else args.segment_net_file,
            #  is_depth=True, 
            #  is_idle_action=False,
             is_ds4_oracle=is_ds4_oracle,
             is_visualizer=not args.no_visual,
             is_visualizer_blocking=not is_ds4_oracle, 
             verbose=args.verbose,
             depth_gaussion_noise_scale=args.noise_depth,
            #  is_dummy=False,
)
print(env.observation_space)
env.seed = args.s

env.reset()
for _ in range(args.repeat):
    env.reset()
    
    # sleep(2)
    done =False
    while not done:
        # action = 0
        # action = env.get_oracle_action(noise_scale=0.15)
        # action = env.get_oracle_action(noise_scale=0.0)
        # action = env.action_space.sample()
        if args.action == 'random': 
            action = env.action_space.sample()
        elif args.action == 'oracle':
            action = env.get_oracle_action()
            if action ==9:
                break
        else:
            action = int(args.action)
        obs, reward, done, info = env.step(action)
        print_dict = {}
        if "image" in obs.keys():
            obs["image"] = obs["image"].shape

        obs.pop("depth_xyz", None)
        print_dict.update({"REWARD": reward})
        print_dict.update({"Done": done})
        print_dict.update({"OBS-"+k:v for k,v in obs.items()})
        print_dict.update({"INFO-"+k:v for k,v in info.items()})
        
        # print(env.get_oracle_action()"" 

        print(f"========= step {env.timestep} ===============")
        print(" \n".join(["{}: {} ".format(k, v) for k,v in print_dict.items()]))
        print()
        # print(obs.keys())
        # print(obs['image'].dtype)
        # print(f"load {sys.getsizeof(obs['image'])}")
        # print(obs['image'].shape)
        # print(obs['image'].nbytes)
        # print(f"item size {obs['image'].itemsize}")
        # print(action)
        if (not args.no_visual):
            env.render(mode=render_mode)
            if not env.is_active:
                break
        
        
        # obs_str = ["|{}: {} ".format(str(k),str(v)) for k,v in obs.items()]
        # obs_str = ' '.join(obs_str)
        # print(obs_str)
        # print(f'steps: {env.timestep}| reward:{reward} | action: {action}, | obs: {obs.keys()}')

    print("=======finish=========")
    print(info)

print('press ctrl + c to exit')
# spin()
env.close()

