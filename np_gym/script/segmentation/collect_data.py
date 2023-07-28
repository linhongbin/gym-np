from gym_np.env.wrapper import make_env
from gym_np.tool.segment import SegmentEngine
from tqdm import tqdm
import cv2
import pathlib
import argparse
from time import sleep
parser = argparse.ArgumentParser()
parser.add_argument('--fill', type=int, default=100)
parser.add_argument('--root-path', type=str, default="./data/seg/origin/rgb/")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--robot', type=str, default='ambf') # [ambf, dvrk]
parser.add_argument('--platform', type=str, default='phantom') #[cuboid, phantom]
parser.add_argument('--action', type=str, default='random') # [random, zero, oracle]
parser.add_argument('--arm', type=str, default='psm2') # [psm1, psm2]
parser.add_argument('--preprocess-type', type=str, default='segment_script') # [segment_net, segment_script]
parser.add_argument('--image-type', type=str, default='origin_rgb') #[zoom_needle_gripper_boximage, zoom_needle_boximage]
parser.add_argument('--clutch', type=int, default=-1)
parser.add_argument('--timelimit', type=int, default=100)
# parser.add_argument('--render-mode', type=str, default="human") # [human_depth, human]

parser.add_argument('--no-visual', action='store_true')
# parser.add_argument('--ds4', action='store_true')
# parser.add_argument('--idle-action', action='store_true')
parser.add_argument('--resize', action='store_true')
parser.add_argument('--sanity-check-file', type=str, default='none')


args = parser.parse_args()

if args.sanity_check_file is not 'none':
    seg_engine = SegmentEngine(segment_net_file=args.sanity_check_file, 
                            process_type='segment_net', 
                            robot_type=args.robot, 
                            image_type="zoom_needle_gripper_boximage")
else:
    seg_engine = None

env = make_env(
                robot_type=args.robot,
             platform_type=args.platform, #[cuboid, phantom]
             preprocess_type=args.preprocess_type, 
             image_type=args.image_type,
             scalar2image_obs_key=[],
             action_arm_device=args.arm,
            reset_needle_mode="manual",
             clutch_start_engaged=args.clutch,
             resize_resolution=64 if args.resize else -1,
             timelimit=args.timelimit, 
             is_depth=True, 
            #  is_idle_action=False,
             is_ds4_oracle=args.robot=="dvrk",
             is_visualizer=False,
            #  is_visualizer_blocking=not args.ds4, 
            #  is_dummy=False,
            is_segfaildone=False,
            is_save_anomaly_pic=False,
)
env.seed = args.seed
_dir = pathlib.Path(args.root_path).expanduser()
_dir = _dir / args.robot / args.arm / args.action / "timelimit{}".format(args.timelimit) /str(args.seed) 
if seg_engine is not None:
    _dir = _dir / "sanity_check"
_dir.mkdir(parents=True, exist_ok=True)
pbar = tqdm(total=args.fill)
cnt=1
eps=0
while cnt<args.fill:
    env.reset()
    done = False
    eps+=1
    while not done and cnt<args.fill:
        if args.action =='oracle':
            if args.robot == 'dvrk':
                action = env.get_oracle_action(env.action_space.n + 1)
            else:
                action = env.get_oracle_action()
        elif args.action =='random':
            action = env.action_space.sample()
        if action == env.action_space.n and args.robot == 'dvrk':
            break
        obs, _, done, _ = env.step(action)
        if seg_engine is not None:
            _, is_sucess = seg_engine.predict_mask(obs["image"])
            if is_sucess:
                continue
            file = f'{args.robot}_{args.arm}_{args.action}_tlim{args.timelimit}_s{args.seed}_eps{eps+1}_step{env.timestep}_sanitycheck.png'
        else:
            # cnt +=1
            file = f'{args.robot}_{args.arm}_{args.action}_tlim{args.timelimit}_s{args.seed}_eps{eps+1}_step{env.timestep}.png'
        cv2.imwrite(str(_dir / file), cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR))
        cnt+=1
        pbar.update(1)
        pbar.set_description(f"{cnt}")
pbar.close()
env.close()
