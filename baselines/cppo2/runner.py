import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, dec_r_coef=0.1):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.dec_states = model.dec_initial_state
        self.dec_r_coef = dec_r_coef

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_obs1, mb_rewards1, mb_rewards2 = [], [], []
        mb_states = self.states
        mb_encs = []
        epinfos = []
        # For n in range number of steps
        self.dec_states = self.model.dec_initial_state
        for _ in range(self.nsteps):
            enc = self.obs[:, -9:]
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs, dec_r, self.dec_states = self.model.step(self.obs, S=self.states, M=self.dones, dec_S=self.dec_states, dec_M=self.dones, dec_Z=enc)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_encs.append(enc)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], _rewards, self.dones, infos = self.env.step(actions)
            mb_obs1.append(self.obs.copy())
            rewards = _rewards + dec_r * self.dec_r_coef
            mb_rewards1.append(_rewards)
            mb_rewards2.append(dec_r * self.dec_r_coef)
            mb_rewards.append(rewards)
            # print(_rewards, dec_r * self.dec_r_coef, rewards)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
        #batch of steps to batch of rollouts
        r1 = np.mean(mb_rewards1)
        r2 = np.mean(mb_rewards2)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs1 = np.asarray(mb_obs1, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_encs = np.asarray(mb_encs)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        check(mb_encs, mb_dones)
        return (*map(sf01, (mb_obs, mb_obs1, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_encs)),
            mb_states, epinfos, r1, r2)

def check(encs, dones):
    n, nenv, c = encs.shape
    nd, nenvd = dones.shape
    assert(n==nd and nenv==nenvd)
    for i in range(n-1):
        for j in range(nenv):
            if dones[i,j] == False:
                try:
                    assert(np.array_equal(encs[i,j,:],encs[i+1,j,:])==True)
                except:
                    print(i, j, dones[i,j], encs[i,j,:], encs[i+1,j,:])
                    import pdb; pdb.set_trace()
            if np.array_equal(encs[i,j,:],encs[i+1,j,:])==False:
                try:
                    assert(dones[i,j] == True)
                except:
                    print(i, j, dones[i,j], encs[i,j,:], encs[i+1,j,:])
                    import pdb; pdb.set_trace()


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


