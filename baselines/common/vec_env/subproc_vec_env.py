import multiprocessing as mp

import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'get_observation_space1':
                ob_space = env.observation_space1
                remote.send((ob_space))
            elif cmd == 'get_obs':
                obs = env.get_obs()
                remote.send((obs))
            elif cmd == 'set_maze_sample':
                result = env.set_maze_sample(data)
                remote.send((result))
            elif cmd == 'get_task_state':
                result = env.get_task_state()
                remote.send((result))
            elif cmd == 'set_task_state':
                try:
                    result = env.set_task_state(*data)
                except:
                    import pdb; pdb.set_trace()
                remote.send((result))
            elif cmd == 'get_task_name':
                name = env.get_task_name()
                remote.send((name))
            elif cmd == 'get_task_enc':
                name = env.get_current_enc()
                remote.send((name))
            elif cmd == 'get_max_task_name':
                name = env.get_max_task_name()
                remote.send((name))
            elif cmd == 'get_task_num':
                num = env.get_task_num()
                remote.send((num))
            elif cmd == 'next_task':
                num = env.next_task()
            elif cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def get_obs(self):
        mb_obs = None
        for i in range(len(self.remotes)):
            self.remotes[i].send(('get_obs', None))
            obs = self.remotes[i].recv()
            obs = np.expand_dims(obs, axis=0)
            if mb_obs is None:
                mb_obs = obs
            else:
                mb_obs = np.concatenate((mb_obs, obs), axis=0)
        return mb_obs

    @property
    def observation_space1(self):
        self.remotes[0].send(('get_observation_space1', None))
        observation_space1 = self.remotes[0].recv()
        return observation_space1

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        try:
            obs, rews, dones, infos = zip(*results)
        except:
            print(results)
            import pdb; pdb.set_trace()
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    @property
    def task_num(self):
        self.get_task_num_async()
        return self.get_task_num_wait()

    def get_task_num_async(self):
        self._assert_not_closed()
        self.remotes[0].send(("get_task_num", None))
        self.waiting = True

    def get_task_num_wait(self):
        self._assert_not_closed()
        results = self.remotes[0].recv()
        self.waiting = False
        return results

    @property
    def max_task_name(self):
        self.get_max_task_name_async()
        return self.get_max_task_name_wait()

    def get_max_task_name_async(self):
        self._assert_not_closed()
        self.remotes[0].send(("get_max_task_name", None))
        self.waiting = True

    def get_max_task_name_wait(self):
        self._assert_not_closed()
        result = self.remotes[0].recv()
        self.waiting = False
        return result

    @property
    def task_enc(self):
        self.get_task_enc_async()
        return self.get_task_enc_wait()

    def get_task_enc_async(self):
        self._assert_not_closed()
        for remote in self.remotes:
           remote.send(("get_task_enc", None)) 
        self.waiting = True
    
    def get_task_enc_wait(self):
        self._assert_not_closed()
        results = np.asarray([remote.recv() for remote in self.remotes])
        self.waiting = False
        return results

    def set_maze_sample(self, state):
        self.set_maze_sample_async(state)
        return self.set_maze_sample_wait()

    def set_maze_sample_async(self, state):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(("set_maze_sample", state))
        self.waiting = True

    def set_maze_sample_wait(self):
        self._assert_not_closed()
        for remote in self.remotes:
            result = remote.recv()
            assert(result==True)
        self.waiting = False

    def get_task_state(self):
        self.get_task_state_async()
        return self.get_task_state_wait()

    def get_task_state_async(self):
        self._assert_not_closed()
        for i in range(len(self.remotes)):
            self.remotes[i].send(("get_task_state", None))
        self.waiting = True

    def get_task_state_wait(self):
        self._assert_not_closed()
        rob_state_list = []
        for remote in self.remotes:
            (qpos, qvel, e_id) = remote.recv()
            rob_state_list.append((qpos, qvel, e_id))
        self.waiting = False
        return rob_state_list

    def set_task_state(self, batch_rob_s):
        self.set_task_state_async(batch_rob_s)
        return self.set_task_state_wait()

    def set_task_state_async(self, batch_rob_s):
        self._assert_not_closed()
        assert(len(batch_rob_s)==len(self.remotes))
        for i in range(len(self.remotes)):
            self.remotes[i].send(("set_task_state", batch_rob_s[i]))
        self.waiting = True

    def set_task_state_wait(self):
        self._assert_not_closed()
        for remote in self.remotes:
            result = remote.recv()
            assert(result == True)
        self.waiting = False
        return True

    @property
    def task_name(self):
        self.get_task_name_async()
        return self.get_task_name_wait()

    def get_task_name_async(self):
        self._assert_not_closed()
        self.remotes[0].send(("get_task_name", None))
        self.waiting = True

    def get_task_name_wait(self):
        self._assert_not_closed()
        result = self.remotes[0].recv()
        self.waiting = False
        return result

    def next_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(("next_task", None))

    def reset_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_task', None))

    def next_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('next_task', None))

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return _flatten_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)
