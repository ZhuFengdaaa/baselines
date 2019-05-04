import random
import numpy as np


class Memory():
    def __init__(self, size=10000, clip_size=None, nenv=1, batch_size=None):
        self.m = []
        self.size = size
        self.clip_size=clip_size
        self.nenv = nenv
        if batch_size is None:
            self.batch_size=nenv
        else:
            self.batch_size=batch_size

    def add_memory(self, episode):
        if len(self.m) < self.size:
            self.m.append(episode)
        else:
            if random.uniform(0, 1) > 0.5:
                idx = random.randint(0, self.size-1)
                self.m[idx] = episode

    def retrieve_memory(self, idx=None, _pos=None):
        if idx is None:
            idx = random.randint(0, len(self.m)-1)
        if self.clip_size is None or self.clip_size>=self.m[idx][0].shape[0]:
            return self.m[idx]
        else:
            (obs, dec_Z, masks) = self.m[idx]
            l = obs.shape[0]
            if _pos is None:
                _pos = random.randint(0, l-1-self.clip_size)
            return (obs[_pos:_pos+self.clip_size,:], dec_Z, masks[_pos:_pos+self.clip_size])

    def set(self, episodes):
        (obs, dec_Z, masks) = episodes
        nsteps = obs.shape[0] // self.nenv
        nc = obs.shape[1]
        obs = np.reshape(obs, (self.nenv, nsteps, -1))
        masks = np.reshape(masks, (self.nenv, nsteps))
        assert obs.shape[2] == nc
        for i in range(self.nenv):
            episode = (obs[i, :, :], dec_Z[i], masks[i,:])
            self.add_memory(episode)

    def get(self):
        batch_episode=None
        batch_dec_Z=None
        batch_dec_M=None
        for i in range(self.batch_size):
            episode, dec_Z, dec_M = self.retrieve_memory()
            episode = np.expand_dims(episode, 0)
            dec_Z = np.expand_dims(dec_Z, 0)
            dec_M = np.expand_dims(dec_M, 0)
            if batch_episode is None and batch_dec_Z is None and batch_dec_M is None:
                batch_episode = episode
                batch_dec_Z = dec_Z
                batch_dec_M = dec_M
            else:
                batch_episode = np.concatenate((batch_episode, episode), axis=0)
                batch_dec_Z = np.concatenate((batch_dec_Z, dec_Z), axis=0)
                batch_dec_M = np.concatenate((batch_dec_M, dec_M), axis=0)
        batch_dec_Z = np.expand_dims(batch_dec_Z, axis=1)
        batch_dec_Z = np.repeat(batch_dec_Z, batch_episode.shape[1],axis=1)
        batch_episode = batch_episode.reshape(-1, batch_episode.shape[2])
        batch_dec_Z = batch_dec_Z.reshape(-1, batch_dec_Z.shape[2])
        batch_dec_M = batch_dec_M.reshape(-1)
        assert batch_episode.shape[0] == batch_dec_Z.shape[0]
        return batch_episode, batch_dec_Z, batch_dec_M
