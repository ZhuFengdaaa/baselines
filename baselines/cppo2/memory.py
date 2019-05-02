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
        if self.clip_size is None or self.clip_size>=self.m[idx][0].shape[1]:
            return self.m[idx]
        else:
            episode = self.m[idx]
            l = episode[0].shape[1]
            if _pos is None:
                _pos = random.randint(0, l-1-self.clip_size)
            return (episode[0][_pos:_pos+self.clip_size,:], episode[1])

    def set(self, episodes):
        nsteps = episodes[0].shape[0] // self.nenv
        nc = episodes[0].shape[1]
        episodes = (np.reshape(episodes[0], (self.nenv, nsteps, -1)), episodes[1])
        assert episodes[0].shape[2] == nc
        for i in range(self.nenv):
            episode = (episodes[0][i, :, :], episodes[1][i])
            self.add_memory(episode)

    def get(self):
        batch_episode=[]
        batch_dec_Z=[]
        for i in range(self.batch_size):
            episode, dec_Z = self.retrieve_memory()
            batch_episode.append(episode)
            batch_dec_Z.append(dec_Z)
        return np.asarray(batch_episode), np.asarray(batch_dec_Z)
