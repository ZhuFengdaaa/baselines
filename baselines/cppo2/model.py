import tensorflow as tf
import functools
import numpy as np

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.cppo2.runner import Runner
from .decoder import Decoder
from .memory import Memory, EpiMemory

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ob_space1, ac_space, enc_space, nbatch_act, nbatch_train,
            nsteps, ent_coef, vf_coef, sf_coef, max_grad_norm, microbatch_size=None, nenv=1,nsteps_dec=100,  nsteps_concat=100, dec_batch_size=3200, concat_batch_size=3200, concat_coef=0, env2=None, gamma=None, lam=None):
        self.sess = sess = get_session()
        self.dec_m = Memory(clip_size=nsteps_dec, nenv=nenv, batch_size=dec_batch_size//nsteps_dec)
        self.concat_nenvs = concat_nenvs = concat_batch_size//nsteps_concat
        self.epi_m = EpiMemory(clip_size=nsteps_concat, batch_size=concat_nenvs)
        self.env2 = env2
        self.gamma = gamma
        self.nsteps_concat = nsteps_concat
        self.ob_space1 = ob_space1.shape[0]

        with tf.variable_scope('cppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)
            self.concat_act_model = concat_model_act = policy(concat_nenvs, 1, sess)
            self.concat_train_model = concat_model_train = policy(concat_batch_size, nsteps_concat, sess)

        with tf.variable_scope('dec_model', reuse=tf.AUTO_REUSE):
            nbatch_train_dec = nenv * nsteps_dec
            self.act_dec = Decoder(nbatch_act, 1, sess, ob_space1, enc_space)
            self.train_dec = Decoder(dec_batch_size, nsteps_dec, sess, ob_space1, enc_space)
            self.batch_act_dec = Decoder(concat_batch_size, nsteps_concat, sess, ob_space1, enc_space)
            self.concat_act_dec = Decoder(concat_nenvs, 1, sess, ob_space1, enc_space, X=self.concat_act_model.sf)
            self.concat_train_dec = Decoder(concat_batch_size, nsteps_concat, sess, ob_space1, enc_space, X=self.concat_train_model.sf)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        self.M = M = tf.placeholder(tf.float32, [None])
        self.concat_mask = concat_mask = tf.placeholder(tf.float32, [None])
        self.S = S = tf.placeholder(tf.float32, [None, self.ob_space1])
        self.h_label = h_label = tf.placeholder(tf.float32, [None, 128])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        spred = train_model.sf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)
        sf_losses = tf.reduce_mean(tf.abs(spred - S), axis=1) * (1-M)
        print(S, spred, tf.abs(spred - S), M)
        sf_loss = tf.reduce_mean(sf_losses) * sf_coef

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + sf_loss

        # dec loss
        self._dec_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_dec.dec_Z, logits=self.train_dec.h1)
        self._dec_losses2 = tf.clip_by_value(self._dec_loss,-5. , 5.)
        self.dec_loss = tf.reduce_mean(self._dec_losses2)

        # feature loss
        h_diff = h_label-self.concat_train_dec.h
        self.h_diff_square = tf.reduce_sum(h_diff*h_diff, 1)
        self.h_diff_square_mask = tf.math.sqrt(tf.reduce_sum(h_diff*h_diff, 1)) * (1-self.concat_mask)
        self.h_loss = tf.reduce_sum(self.h_diff_square_mask) * concat_coef
        self.end2end_div_loss = - self.concat_train_dec.r * 10
        self.h_loss2 = self.h_loss + self.end2end_div_loss
        self.pt_op1 = tf.print(self.h_diff_square)
        self.pt_op2 = tf.print(self.h_diff_square_mask)
        self.pt_op3 = tf.print([self.h_loss, self.end2end_div_loss])

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('cppo2_model')
        dec_params = tf.trainable_variables('dec_model')
        concat_params = params
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        self.dec_grads_and_var = self.trainer.compute_gradients(self.dec_loss, dec_params)
        concat_grads_and_var = self.trainer.compute_gradients(self.h_loss2, concat_params)

        grads, var = zip(*grads_and_var)
        self.dec_grads, dec_var = zip(*self.dec_grads_and_var)
        concat_grads, concat_var = zip(*concat_grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            concat_grads, concat_grad_norm = tf.clip_by_global_norm(concat_grads, max_grad_norm)
            # self.dec_global_norm always inf, I dont know why
            self.dec_global_norm = tf.linalg.global_norm(self.dec_grads)
            self.pt = tf.print(self.dec_global_norm)
            self.dec_grads2, dec_grad_norm = tf.clip_by_global_norm(self.dec_grads, 1.5)
        grads_and_var = list(zip(grads, var))
        self.dec_grads_and_var2 = list(zip(self.dec_grads2, dec_var))
        concat_grads_and_var = list(zip(concat_grads, concat_var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.concat_train_op = self.trainer.apply_gradients(concat_grads_and_var)
        self.dec_train_op = self.trainer.apply_gradients(self.dec_grads_and_var2)
        self.loss_names = ['policy_loss', 'value_loss', 'state_loss'
                , 'policy_entropy', 'approxkl', 'clipfrac']
        self.dec_loss_names = ['dec_loss']
        self.concat_loss_names = ['concat_loss']
        self.stats_list = [pg_loss, vf_loss, sf_loss
                , entropy, approxkl, clipfrac]
        self.dec_stats_list = [self.dec_loss]


        self.train_model = train_model
        self.act_model = act_model
        # self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.dec_initial_state = self.act_dec.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def step(self, observation, **extra_feed):
        actions, values, spred, states, neglogpacs = self.act_model.step(observation, **extra_feed)
        dec_r, dec_states = self.act_dec.step(observation[:, :self.ob_space1], **extra_feed)
        return actions, values, spred, states, neglogpacs, dec_r, dec_states

    def train(self, lr, cliprange, obs, obs1, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.S : obs1[:,:self.ob_space1],
            self.M : masks
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        policy_loss = self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

        return policy_loss

    def dec_train(self, dec_lr, obs, masks, decs):
        self.dec_m.set((obs, decs, masks))
        batch_episode, batch_dec_Z, batch_dec_M =self.dec_m.get()
        dec_S = self.train_dec.initial_state
        dec_map = {
            self.train_dec.X : batch_episode[:,:self.ob_space1],
            self.train_dec.dec_Z : batch_dec_Z,
            self.train_dec.dec_S : dec_S,
            self.train_dec.dec_M : batch_dec_M,
            self.LR : dec_lr
        }
        dg, dgn = self.sess.run([self.dec_grads, self.dec_global_norm], dec_map)
        try:
            return self.sess.run(
                self.dec_stats_list + [self.dec_train_op] + [],
                dec_map
            )[:-1]
        except Exception as e:
            print("if global norm inf or nan, pass", e)
            pass
            # print(e)
            # import pdb; pdb.set_trace()
        return [0]

    def cal_R(self, rs, masks):
        rs = rs.reshape(-1)
        masks = masks.reshape(-1)
        assert(rs.shape[0]==masks.shape[0])
        R=0
        for i in reversed(range(rs.shape[0])):
            R = R * self.gamma
            R = R+rs[i]*(1-masks[i])
        return R

    def mask_out(self,dones):
        n, steps = dones.shape
        for i in range(n):
            flag=False
            for j in range(steps):
                if dones[i,j] == True:
                    flag=True
                if flag == True:
                    dones[i,j]=True
        return dones

    def concat_train(self, concat_lr, epis):
        self.epi_m.set(epis)
        batch_s, batch_r, batch_m, batch_rob_s, batch_idx = self.epi_m.get()
        batch_s0 = batch_s[:, 0, :]
        self.env2.reset()
        self.env2.set_task_state(batch_rob_s)
        # self.env2.step()
        assert(batch_r.shape[0] == batch_s.shape[0] == batch_m.shape[0]==len(batch_rob_s)==self.concat_nenvs)
        assert(batch_r.shape[1]==batch_s.shape[1]==batch_m.shape[1]==self.nsteps_concat)
        dec_state = self.concat_act_dec.initial_state
        _batch_s0 = self.env2.get_obs()
        # assert(np.array_equal(batch_s0, _batch_s0))
        dec_z = []
        l_z = self.env2.task_num
        for i in range(len(batch_rob_s)):
            z = [0 for j in range(l_z)]
            z[batch_rob_s[i][2]] = 1
            dec_z.append(z)
        dec_z = np.asarray(dec_z)
        dec_z = np.expand_dims(dec_z, 1)
        dec_z = np.tile(dec_z, (1, self.nsteps_concat, 1))
        dec_z = dec_z.reshape(-1, dec_z.shape[2])
        batch_st = batch_s0
        mb_obs=[]
        mb_actions=[]
        mb_rewards=[]
        mb_dones=[]
        mb_rob_states = []
        dones = [False for _ in range(self.concat_nenvs)]
        for i in range(self.nsteps_concat):
            step_map = {
                self.concat_act_model.X : batch_st,
                self.concat_act_dec.dec_S : dec_state,
                self.concat_act_dec.dec_M : batch_m[:, i]
            }
            actions, h_label, dec_state = self.sess.run(
                [self.concat_act_model.action, self.concat_act_dec.h, self.concat_act_dec.snew],
                step_map
            )
            mb_obs.append(batch_st)
            mb_actions.append(actions)
            rob_state = self.env2.get_task_state()
            mb_rob_states.append(rob_state)
            mb_dones.append(dones)
            batch_st, rewards, dones, infos = self.env2.step(actions)
            mb_rewards.append(rewards)
        mb_obs=np.asarray(mb_obs).swapaxes(0,1)
        mb_actions=np.asarray(mb_actions).swapaxes(0,1)
        mb_rewards=np.asarray(mb_rewards).swapaxes(0,1)
        mb_dones=np.asarray(mb_dones).swapaxes(0,1)
        mb_dones=self.mask_out(mb_dones)

        dec_state = self.batch_act_dec.initial_state
        _batch_s = batch_s.reshape(-1, batch_s.shape[2])
        _batch_m = batch_m.reshape(-1)
        label_map = {
            self.batch_act_dec.X : _batch_s[:,:self.ob_space1],
            self.batch_act_dec.dec_S : dec_state,
            self.batch_act_dec.dec_M : _batch_m
        }
        h_labels = self.sess.run(
            [self.batch_act_dec.h],
            label_map
        )[0]

        # obs, obs1, spred, returns, rewards, masks, masks1, actions, values, neglogpacs, encs, states, epinfos, r1, r2, epis = self.concat_runner.run()
        assert(batch_r.shape==mb_rewards.shape)
        train_masks = np.full(batch_m.shape, False)
        for i in range(self.concat_nenvs):
            R1 = self.cal_R(batch_r[i,:], batch_m[i,:])
            R2 = self.cal_R(mb_rewards[i,:], mb_dones[i,:])
            print(R1, R2)
            if R1 > R2: # TODO margin
                train_masks[i,:] = np.logical_or(batch_m[i,:].astype(bool), mb_dones[i,:].astype(bool))
            else:
                # R1 < R2, replace memory
                train_masks[i,:] = True
                for j in range(self.nsteps_concat):
                    if mb_dones[i,j] == True:
                        mb_r_s = []
                        for _j in range(j):
                            mb_r_s.append(mb_rob_states[_j][i])
                        epi = {
                            "a":mb_actions[i,:j],
                            "s":mb_obs[i,:j],
                            "r":mb_rewards[i,:j],
                            "rob_s":mb_r_s
                        }
                        self.epi_m.set_epi(batch_idx[i], epi)
                        break
        train_masks = train_masks.reshape(-1)
        mb_obs1 = mb_obs.reshape(-1, mb_obs.shape[2])
        self.env2.reset()
        self.env2.set_task_state(batch_rob_s)
        dec_state = self.concat_train_dec.initial_state
        train_map = {
            self.concat_train_model.X : mb_obs1,
            self.concat_train_dec.dec_S : dec_state,
            self.concat_train_dec.dec_M : train_masks,
            self.concat_train_dec.dec_Z : dec_z,
            self.concat_mask: train_masks,
            self.h_label : h_labels,
            self.LR : concat_lr,
            self.LR : concat_lr,
        }

        h_loss = self.sess.run(
            [self.h_loss, self.concat_train_op, self.pt_op1, self.pt_op2, self.pt_op3],
            train_map
        )[:1]
        return h_loss
