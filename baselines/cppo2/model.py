import tensorflow as tf
import functools
import numpy as np

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from .decoder import Decoder
from .memory import Memory

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
            nsteps, ent_coef, vf_coef, sf_coef, max_grad_norm, microbatch_size=None, nenv=1, nsteps_dec=100, dec_batch_size=3200):
        self.sess = sess = get_session()
        self.dec_m = Memory(clip_size=nsteps_dec, nenv=nenv, batch_size=dec_batch_size//nsteps_dec)
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

        with tf.variable_scope('dec_model', reuse=tf.AUTO_REUSE):
            nbatch_train_dec = nenv * nsteps_dec
            self.act_dec = Decoder(nbatch_act, 1, sess, ob_space1, enc_space)
            self.train_dec = Decoder(dec_batch_size, nsteps_dec, sess, ob_space1, enc_space)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        self.M = M = tf.placeholder(tf.float32, [None])
        self.S = S = tf.placeholder(tf.float32, [None, ob_space.shape[0]])
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
        sf_loss = tf.reduce_mean(sf_losses)

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
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + sf_loss * sf_coef
        self._dec_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_dec.dec_Z, logits=self.train_dec.h1)
        self._dec_losses2 = tf.clip_by_value(self._dec_loss,-5. , 5.)
        self.dec_loss = tf.reduce_mean(self._dec_losses2)

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('cppo2_model')
        dec_params = tf.trainable_variables('dec_model')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        self.dec_grads_and_var = self.trainer.compute_gradients(self.dec_loss, dec_params)

        grads, var = zip(*grads_and_var)
        self.dec_grads, dec_var = zip(*self.dec_grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            # self.dec_global_norm always inf, I dont know why
            self.dec_global_norm = tf.linalg.global_norm(self.dec_grads)
            self.pt = tf.print(self.dec_global_norm)
            self.dec_grads2, dec_grad_norm = tf.clip_by_global_norm(self.dec_grads, 5)
        grads_and_var = list(zip(grads, var))
        self.dec_grads_and_var2 = list(zip(self.dec_grads2, dec_var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.dec_train_op = self.trainer.apply_gradients(self.dec_grads_and_var2)
        self.loss_names = ['policy_loss', 'value_loss', 'state_loss'
                , 'policy_entropy', 'approxkl', 'clipfrac']
        self.dec_loss_names = ['dec_loss']
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
        actions, values, states, neglogpacs = self.act_model.step(observation, **extra_feed)
        dec_r, dec_states = self.act_dec.step(observation[:,:self.ob_space1], **extra_feed)
        return actions, values, states, neglogpacs

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
            self.S : obs1,
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
