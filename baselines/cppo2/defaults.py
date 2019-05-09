import os

def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy',
    )

def maze():
    maze_dict = dict(
        nsteps=2048,
        nminibatches=32,
        num_env=8,
        num_timesteps=1e6,
        save_path='models/default',
        dec_lr=1e-3,
        concat_lr=1e-3,
        dec_r_coef=0.033,
        concat_coef=0.01,
        sf_coef=10,
        nsteps_dec=100,
        dec_batch_size=6400,
        nsteps_concat=100,
        concat_batch_size=6400,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy',
        estimate_s=True
    )
    if not os.path.exists(maze_dict['save_path']):
        os.makedirs(maze_dict['save_path'])
    with open('{}/config.txt'.format(maze_dict['save_path']), 'w+') as f:
        for key, value in maze_dict.items():
            f.writelines('{} = {}\n'.format(key, value))

    return maze_dict


def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )

def retro():
    return atari()
