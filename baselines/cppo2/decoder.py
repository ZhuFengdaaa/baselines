import tensorflow as tf
from baselines.a2c import utils
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape


class Decoder():
    def __init__(self, nbatch, nsteps, sess, ob_space, enc_space, X=None, nlstm=128, layer_norm=False):
        self.sess = sess or tf.get_default_session()
        self.X = X if X is not None else observation_placeholder(ob_space, batch_size=nbatch)
        nenv = nbatch // nsteps
        self.dec_Z = tf.placeholder(tf.float32, [nbatch, enc_space])
        # _h = tf.concat([self.X, self.dec_Z], 1)
        h = tf.layers.flatten(self.X)
        self.dec_M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        self.dec_S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) # states
        z_prob = tf.fill([nbatch, enc_space], 1.0/enc_space)
        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(self.dec_M, nenv, nsteps)
        if layer_norm:
            h5, self.snew = utils.lnlstm(xs, ms, self.dec_S, scope='lnlstm', nh=nlstm)
        else:
            h5, self.snew = utils.lstm(xs, ms, self.dec_S, scope='lstm', nh=nlstm)
            # h5, self.snew = utils.lstm(xs, ms, self.dec_S, scope='lstm', nh=nlstm)
        h = seq_to_batch(h5)
        self.h1 = fc(h, 'fc1', nh=enc_space, init_scale=np.sqrt(2))
        logq = self.dec_Z * tf.math.log(tf.nn.softmax(self.h1))
        logp = self.dec_Z * tf.math.log(z_prob)
        self.r = tf.reduce_sum(logq-logp, 1)
        # eheck shape
        self.initial_state = np.zeros(self.dec_S.shape.as_list(), dtype=float)
        # feed_dic = {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        dec_r, dec_state = self._evaluate([self.r, self.snew], observation, **extra_feed)
        return dec_r, dec_state
