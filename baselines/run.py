import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.cppo2.defaults import maze

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    total_timesteps = int(args.num_timesteps)  # fix test bug
    alg_kwargs['num_timesteps'] = total_timesteps

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=maze()['save_path'])  # guanghuixu
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args, extra_args)

    # fix test bug
    if not args.play:
        if hasattr(args, "save_path") and args.save_path is not None and rank == 0:
            save_path = osp.expanduser(args.save_path)
            model.save(save_path)
    task_count = 0
    if args.play:
        task_count += 1
        # if args.maze_sample != True: # fixed test bug
        env.set_maze_sample(False)
        logger.log("Running trained model")
        if hasattr(env.envs[0], "reset_task"):
            env.reset_task()
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = 0
        episode_rew_cnt = 0
        # max_episode = 50
        max_episode = 5
        cnt_episode = 1
        task_id = 0
        flag=True
        nsd_dic = {}
        succeed = {}
        dec_states = None
        enc = env.task_enc
        res_file = open("res_file3.txt", "w")
        entropys = []
        while True:
            if flag==True:
                res_file.write("# %d %d\n" % (task_id, cnt_episode))
                flag=False
            x, y = env.envs[0].wrapped_env.get_body_com("torso")[:2]
            _x, _y = env.envs[0].normalize(x,y)
            res_file.write("%f %f\n" % (_x, _y))
            if args.alg=="cppo2":
                if dec_states is None:
                    dec_states = model.dec_initial_state
                if state is not None:
                    actions, values, state, neglogpacs = model.step(obs,S=state, M=dones, dec_S=dec_states, dec_M=dones, dec_Z=enc)
                else:
                    # actions, _, state, _ = model.step(obs, S=state, M=dones, dec_S=dec_states, dec_M=dones, dec_Z=enc)
                    # train decoder
                    actions, values, state, neglogpacs, dec_r, dec_states, entropy = model.step(obs,S=state, M=dones, dec_S=dec_states, dec_M=dones, dec_Z=enc)
                    # actions, _, _, _ = model.step(obs)
                    entropys.append(entropy)
            else:
                if state is not None:
                    actions, _, state, _ = model.step(obs,S=state, M=dones)
                else:
                    actions, _, _, _ = model.step(obs)

            obs, rew, done, info = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            episode_rew_cnt +=1
            if args.render:
                env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                info = info[0]
                dec_states = None
                print('episode_rew={}'.format(episode_rew/episode_rew_cnt))
                shorten_dist = max(0, env.shorten_dist)
                print('# {}: dist={}'.format(cnt_episode, shorten_dist))
                cnt_episode += 1
                if env.max_task_name not in nsd_dic.keys():
                    nsd_dic[env.max_task_name] = []
                    succeed[env.max_task_name] = []
                nsd_dic[env.max_task_name].append(shorten_dist)
                if "succeed" in info and info["succeed"] == 1:
                    succeed[env.max_task_name].append(1)
                else:
                    succeed[env.max_task_name].append(0)
                if cnt_episode > max_episode:
                    cnt_episode = 1
                    task_id += 1
                    if hasattr(env, "next_task"):
                        if env.next_task() is False:
                            break
                obs = env.reset()
                flag=True
        print(nsd_dic)
        task_nsd = []
        task_spl = []
        shortest_len = {
                "line200": 8,
                "corner00": 8,
                "corner10": 16,
                "empty00": 8,
                "empty10": 12,
                "maze100": 16,
                "maze200": 16,
                "maze210": 16
                }
        for k,v in nsd_dic.items():
            assert(type(v) == list)
            nsd = sum(v)/len(v)
            spl = sum(i[0] * i[1] for i in zip(nsd_dic[k], succeed[k]))/len(v)/shortest_len[k]
            task_nsd.append(nsd)
            task_spl.append(spl)
            print("{}: nsd {} spl {}".format(k, nsd, spl))
        nsd = sum(task_nsd)/len(task_nsd)
        spl = sum(task_spl)/len(task_spl)
        print("total nsd {} spl {}".format(nsd, spl))

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
