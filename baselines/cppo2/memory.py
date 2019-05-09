import random
import numpy as np


class Memory():
    def __init__(self, size=10000, clip_size=None, nenv=1, batch_size=None):
        self.m = []
        self.size = size
        self.clip_size=clip_size
        self.nenv = nenv
        self._n = 8.*1e6/8/2048*10*9
        self._m = size
        if batch_size is None:
            self.batch_size=nenv
        else:
            self.batch_size=batch_size

    def add_memory(self, episode):
        if len(self.m) < self.size:
            self.m.append(episode)
        else:
            if random.uniform(0, 1) < self._m/self._n:
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
            return (obs[_pos:_pos+self.clip_size,:], dec_Z[_pos:_pos+self.clip_size, :], masks[_pos:_pos+self.clip_size])

    def set(self, episodes):
        (obs, encs, masks) = episodes
        nsteps = obs.shape[0] // self.nenv
        nc = obs.shape[1]
        obs = np.reshape(obs, (self.nenv, nsteps, -1))
        encs = np.reshape(encs, (self.nenv, nsteps, -1))
        masks = np.reshape(masks, (self.nenv, nsteps))
        assert obs.shape[2] == nc
        for i in range(self.nenv):
            episode = (obs[i, :, :], encs[i,:, :], masks[i,:])
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
        batch_episode = batch_episode.reshape(-1, batch_episode.shape[2])
        batch_dec_Z = batch_dec_Z.reshape(-1, batch_dec_Z.shape[2])
        batch_dec_M = batch_dec_M.reshape(-1)
        assert batch_episode.shape[0] == batch_dec_Z.shape[0] == batch_dec_M.shape[0]
        return batch_episode, batch_dec_Z, batch_dec_M

class EpiMemory():
    def __init__(self, size=10000, clip_size=100, batch_size=800):
        self.m=[]
        self.size = size
        self.clip_size=clip_size
        self.batch_size = batch_size

    def set_epi(self, idx, epi):
        self.m[idx]=epi

    def set(self, epis):
        for epi in epis:
            if len(self.m) < self.size:
                self.m.append(epi)
            else:
                _idx = random.randint(0, self.size-1)
                if _idx < self.size:
                    self.m[_idx] = epi

    def retrive_memory(self, idx=None, _pos=None):
        if idx is None:
            idx = random.randint(0, len(self.m)-1)
        s = self.m[idx]["s"]
        r = self.m[idx]["r"]
        assert(s.shape[0]==r.shape[0])
        if self.clip_size > s.shape[0]:
            s = np.concatenate((s, np.ones((self.clip_size - s.shape[0], s.shape[1]))), axis=0)
            r = np.concatenate((r, np.ones((self.clip_size - r.shape[0]))), axis=0)
            m = np.concatenate((np.zeros((s.shape[0])), np.ones((self.clip_size - s.shape[0]))), axis=0)
            rob_s = self.m[idx]["rob_s"][0]
        else:
            if _pos is None:
                _pos = random.randint(0, s.shape[0]-self.clip_size)
            s = s[_pos:_pos+self.clip_size,:]
            r = r[_pos:_pos+self.clip_size]
            m = np.zeros((self.clip_size))
            rob_s = self.m[idx]["rob_s"][_pos]
        return s, r, m, rob_s, idx
        
    def get(self):
        batch_s = None
        batch_r = None
        batch_m = None
        batch_rob_s = []
        batch_idx = []
        for i in range(self.batch_size):
            s, r, m, rob_s, idx = self.retrive_memory()
            s = np.expand_dims(s, 0)
            r = np.expand_dims(r, 0)
            m = np.expand_dims(m, 0)
            if i == 0:
                batch_s = s
                batch_r = r
                batch_m = m
            else:
                batch_s = np.concatenate((batch_s, s), axis=0)
                batch_r = np.concatenate((batch_r, r), axis=0)
                batch_m = np.concatenate((batch_m, m), axis=0)
            batch_rob_s.append(rob_s)
            batch_idx.append(idx)
        return batch_s, batch_r, batch_m, batch_rob_s, batch_idx
