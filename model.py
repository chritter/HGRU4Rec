import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
import numpy as np
import pandas as pd
from os import path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class HGRU4Rec:
  # HGRU4Rec Network
  def __init__(self, sess, session_layers, user_layers, n_epochs=10, batch_size=50, learning_rate=0.001,
               decay=0.96, grad_cap=0, sigma=0, dropout_p_hidden_usr=0.3,
               dropout_p_hidden_ses=0.3, dropout_p_init=0.3, init_as_normal=False,
               reset_after_session=True, loss='top1', hidden_act='tanh', final_act=None,
               session_key='session_id', item_key='item_id', time_key='created_at', user_key='user_id', n_sample=0,
               sample_alpha=0.75, user_propagation_mode='init',
               user_to_output=False, user_to_session_act='tanh', n_items=4, checkpoint_dir='', log_dir=''):
    '''
    Set initial the parameters according to user input
    final_act: Not used as input parameter! output layer activation function
    '''

    # define functions
    self.sess = sess
    self.session_layers = session_layers
    self.user_layers = user_layers
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.dropout_p_hidden_usr = dropout_p_hidden_usr
    self.dropout_p_hidden_ses = dropout_p_hidden_ses
    self.dropout_p_init = dropout_p_init
    self.learning_rate = learning_rate
    self.decay = decay
    self.sigma = sigma
    self.init_as_normal = init_as_normal
    self.reset_after_session = reset_after_session
    self.session_key = session_key
    self.item_key = item_key
    self.time_key = time_key
    self.user_key = user_key
    self.grad_cap = grad_cap

    # custom start
    # should there be an option to change this to false?
    self.is_training = True
    # fixed steps
    self.decay_steps = 1e4
    self.n_items = n_items
    self.log_dir = log_dir
    # custom end

    self.user_propagation_mode = user_propagation_mode
    self.user_to_output = user_to_output

    if hidden_act == 'tanh':
      self.hidden_act = self.tanh
    elif hidden_act == 'relu':
      self.hidden_act = self.relu
    else:
      raise NotImplementedError

    if loss == 'top1':
      # choose final activation: tanh is used
      if final_act == 'linear':
        self.final_activation = self.linear
      elif final_act == 'relu':
        self.final_activation = self.relu
      else:
        self.final_activation = self.tanh
      self.loss_function = self.top1
    else:
      raise NotImplementedError('loss {} not implemented'.format(loss))

    # choose activation function of hidden layers
    if hidden_act == 'relu':
      self.hidden_activation = self.relu
    elif hidden_act == 'tanh':
      self.hidden_activation = self.tanh
    else:
      raise NotImplementedError('hidden activation {} not implemented'.format(hidden_act))

    if user_to_session_act == 'relu':
      self.s_init_act = self.relu
    elif user_to_session_act == 'tanh':
      self.s_init_act = self.tanh
    else:
      raise NotImplementedError('user-to-session activation {} not implemented'.format(hidden_act))

    self.n_sample = n_sample
    self.sample_alpha = sample_alpha

    self.checkpoint_dir = checkpoint_dir
    if not path.isdir(self.checkpoint_dir):
      raise Exception("[!] Checkpoint Dir not found")

    # build model
    self.build_model()

    # init variables
    self.sess.run(tf.global_variables_initializer())

    # use save
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    if self.is_training:
      return

    # # use self.predict_state to hold hidden states during prediction.
    # self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
    # ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #   self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

  ########################ACTIVATION FUNCTIONS#########################
  def linear(self, X):
    return X
  def tanh(self, X):
    return tf.nn.tanh(X)
  def softmax(self, X):
    return tf.nn.softmax(X)
  def softmaxth(self, X):
    return tf.nn.softmax(tf.tanh(X))
  def relu(self, X):
    return tf.nn.relu(X)
  def sigmoid(self, X):
    return tf.nn.sigmoid(X)

  ############################LOSS FUNCTIONS######################
  def top1(self, yhat):
    '''
    TOP1 loss
    :param yhat:
    :return:
    '''
    with tf.name_scope("top1"):
      yhatT = tf.transpose(yhat)
      # operation 1/n sigmoid(r_s_j - r_s_i) + sigmoid(r_s_j)
      # r_s_i is tf.diag_part(yhat)
      term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
      term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
      return tf.reduce_mean(term1 - term2)

  class UserGRUCell4Rec(tf.nn.rnn_cell.MultiRNNCell):
    """
    UserGRU cell for HGRU4Rec
    """

    def __init__(self, cells, state_is_tuple=True, hgru4rec=None):
      '''
      Initialize, TF MultiRNNCell, allow states inputs and outputs as n-tuples (state_is_tuple=True)
      :param cells:
      :param state_is_tuple:
      :param hgru4rec:
      '''
      super(HGRU4Rec.UserGRUCell4Rec, self).__init__(cells, state_is_tuple=state_is_tuple)
      #super().__init__(cells, state_is_tuple=state_is_tuple)
      # hgru4rec object to assign resetting of hidden states later in call
      self.hgru4rec = hgru4rec

    def call(self, inputs, state):
      '''
      Run this multi-layer cell on inputs, starting from state.
      Should take as input the session-level representations
      :param inputs: hidden state from last session of user (self.Hs[-1]) , tuple(self.Hu
      :param state: hidden state of previous user representation with one state per cell (tuple(self.Hu))
      :return: Output and state of GRU_usr
      '''

      cur_state_pos = 0
      cur_inp = inputs
      new_states = []

      # loop over cells of TF MultiRNNCell (self) - is this documented?
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("cell_%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(state, [0, cur_state_pos],
                                        [-1, cell.state_size])
            cur_state_pos += cell.state_size
          # (current) input, current state for cell
          # c_m =??, output o, hidden state h
          o, h = cell(cur_inp, cur_state)

          # reset hidden states of user for specific sessions in batch when start of new session?? and start of new user
          h = tf.where(self.hgru4rec.sstart, h, cur_state, name='sel_hu_1')
          h = tf.where(self.hgru4rec.ustart, tf.zeros(tf.shape(h)), h, name='sel_hu_2')

          new_states.append(h)
          cur_inp = h

      new_states = (tuple(new_states) if self._state_is_tuple else
                    array_ops.concat(new_states, 1))

      return cur_inp, new_states

  def build_model(self):
    """

    :return:
    """
    self.X = tf.placeholder(tf.int32, [self.batch_size], name='input_x')
    self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output_y')
    # hidden layer for session
    self.Hs = [tf.placeholder(tf.float32, [self.batch_size, s_size], name='Hs') for s_size in
                  self.session_layers]
    # hidden layer for user
    self.Hu = [tf.placeholder(tf.float32, [self.batch_size, u_size], name='Hu') for u_size in
                  self.user_layers]
    # for marking the start of a session or a user?
    self.sstart = tf.placeholder(tf.bool, [self.batch_size], name='sstart')
    self.ustart = tf.placeholder(tf.bool, [self.batch_size], name='usstart')

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    # USER GRU
    with tf.variable_scope('user_gru'):
      cells = []
      # create each layer in GRU_usr, with number of units u_size
      for u_size in self.user_layers:
        cell = tf.nn.rnn_cell.GRUCell(u_size, activation=self.hidden_act)
        # applies dropout to GRU cell outuput
        cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden_usr))
      # init class UserGRUCell4Rec to assign cells
      stacked_cell = self.UserGRUCell4Rec(cells, hgru4rec=self)
      # pass hidden state of last session, hidden state of current user representation
      # calculate output and new state of GRU_usr
      output, state = stacked_cell(self.Hs[-1], tuple(self.Hu))
      # (is c_m for session m in paper)
      self.Hu_new = state

    # SESSION GRU
    with tf.variable_scope('session_gru'):

      # initialization of weights
      sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + sum(self.session_layers)))
      if self.init_as_normal:
        initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
      else:
        initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
      # input embedding matrix
      embedding = tf.get_variable('embedding', [self.n_items, self.session_layers[0]], initializer=initializer)
      # output embedding matrix
      softmax_W = tf.get_variable('softmax_w', [self.n_items, self.session_layers[0]], initializer=initializer)
      softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

      cells = []
      # create each layer in GRU_ses, with number of units u_size
      for s_size in self.session_layers:
        cell = tf.nn.rnn_cell.GRUCell(s_size, activation=self.hidden_act)
        cells.append(tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden_ses))
      # here we just apply a TF MultiRNNCell function, in contrast to UserGRUCell4Rec
      stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

      # construct weights of hidden layers
      input_states=[]
      for j in range(len(self.session_layers)):
        # initializie hidden state for each layer of session based on last hidden state of user, with
        # number of units given by session_layer; applies user_to_session activation function, dropout (paper)
        # tf.layers.dense implements W_init*c_m +b_init of Eq. 4 in paper
        h_s_init = tf.layers.dropout(self.s_init_act(tf.layers.dense(self.Hu_new[-1], self.session_layers[j])),
                                   rate=self.dropout_p_init, training=self.is_training,
                                   name='h_s_init_{}'.format(j))
        # sessions in batch which start with new session initialize with h_s_init, else take hidden layer of session
        h_s = tf.where(self.sstart, h_s_init, self.Hs[j], name='sel_hs_1_{}'.format(j))
        # session in batch which start with new user initialize with zeros
        h_s = tf.where(self.ustart, tf.zeros(tf.shape(h_s)), h_s, name='sel_hs_2_{}'.format(j))
        input_states.append(h_s)

      # lookup input embeddings based on input batch X
      inputs = tf.nn.embedding_lookup(embedding, self.X, name='embedding_x')

      # get output and state of GRU_ses, based on embedded X and input states
      output, state = stacked_cell(inputs,
                                   tuple(input_states)
                                   )
      self.Hs_new = state


      if self.is_training:
        '''
        Use other examples of the minibatch as negative samples.
        '''
        # for output layer, get item output embedding
        sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
        sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
        # output * W + b
        logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
        # output activation function
        self.yhat = self.final_activation(logits)
        # TOP1 loss/cost calculation
        self.cost = self.loss_function(self.yhat)
        tf.summary.scalar('cost', self.cost)
      else:
        # for predictions, calculate directly output
        logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
        # save predictions in yhat
        self.yhat = self.final_activation(logits)

    if not self.is_training:
      return

    with tf.name_scope("optimizer"):

      # paper uses AdaGrad

      # implements exponential decay of learning rate with minimum of lr of 1e-5
      self.lr = tf.maximum(1e-5,
                         tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay,
                                                    staircase=True))
      # save lr
      tf.summary.scalar('lr', self.lr)

      # use Adam optimizer instead of AdaGrad as in paper
      optimizer = tf.train.AdamOptimizer(self.lr)

      # get all trainable variables
      tvars = tf.trainable_variables()
      # calculate gradients of variables based on loss
      gvs = optimizer.compute_gradients(self.cost, tvars)

      # gradient clipping to avoid exploding gradients
      if self.grad_cap > 0:
        capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
      else:
        capped_gvs = gvs
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    self.merged = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(path.join(self.log_dir, 'train'), self.sess.graph)

  def preprocess_data(self, data):
      '''
      Calculate index columns for session and users
      :param data:
      :return:
      '''
      # sort by user and time key in order
      data.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
      data.reset_index(drop=True, inplace=True)
      # offset index for sessions
      offset_session = np.r_[0, data.groupby([self.user_key, self.session_key], sort=False).size().cumsum()[:-1]]
      # offset index for users
      user_indptr = np.r_[0, data.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum()[:-1]]
      return user_indptr, offset_session

  def iterate(self, data, offset_sessions, user_indptr, reset_state=True, is_validation=False):
    """
    Training for one epoch.
    Implements User-parallel mini-batches
    :param data:
    :param offset_sessions:
    :param user_indptr:
    :param reset_state:
    :return:
    """

    # variables to manage iterations over users
    # number of all users in training set
    n_users = len(user_indptr)
    offset_users = offset_sessions[user_indptr]
    user_idx_arr = np.arange(n_users - 1)
    # number of users to iterative over for each batch
    user_iters = np.arange(self.batch_size)
    # iterator to be incremented when new users are processed
    user_maxiter = user_iters.max()
    # identifies the start and end for users of first batch
    user_start = offset_users[user_idx_arr[user_iters]]
    user_end = offset_users[user_idx_arr[user_iters] + 1]

    # variables to manage iterations over sessions
    #
    session_iters = user_indptr[user_iters]
    # identifies the start and end of each session for each user
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    # flags for resetting hidden states of sessions and users, done at end of each session and user below
    sstart = np.zeros((self.batch_size,), dtype=np.bool)
    ustart = np.zeros((self.batch_size,), dtype=np.bool)
    finished = False
    n = 0

    # keeps track of costs
    c = []

    summary = None

    # initialize hidden layers for sessions
    Hs_new = [np.zeros([self.batch_size, s_size], dtype=np.float32) for s_size in self.session_layers]
    # initialize hidden layers for users
    Hu_new = [np.zeros([self.batch_size, u_size], dtype=np.float32) for u_size in self.user_layers]

    # continue until no sessions or no users are available
    while not finished:

      # identify how many items in parallel with the the user-parallel session approach can be processed
      session_minlen = (session_end - session_start).min()
      out_idx = data.ItemIdx.values[session_start]

      # loop over batches, until any of user-parallel sessions is empty/no further items are in session
      for i in range(session_minlen - 1):

        # indices of input items X (one item per user) and output items Y as indices for training
        # output items from last item becomes input for next item
        in_idx = out_idx
        out_idx = data.ItemIdx.values[session_start + i + 1]

        #if self.n_sample:
          #   sample = self.neg_sampler.next_sample()
          #   y = np.hstack([out_idx, sample])
          # else:
        y = out_idx

        # prepare input for NN
        feed_dict = {self.X: in_idx, self.Y: y, self.sstart: sstart, self.ustart: ustart}
        # pass initialized hidden states to input for NN
        for j in range(len(self.Hs)):
          feed_dict[self.Hs[j]] = Hs_new[j]
        for j in range(len(self.Hu)):
          feed_dict[self.Hu[j]] = Hu_new[j]

        # define which tensors to compute
        fetches = []
        if is_validation == False:
          # merged is summary, need to return hidden states for sessions and users
          fetches = [self.merged, self.cost, self.Hs_new, self.Hu_new, self.global_step, self.lr, self.train_op]
        else:
          fetches = [self.merged, self.cost, self.Hs_new, self.Hu_new]

        # run the network with input, fetches
        summary, cost, Hs_new, Hu_new, step, lr, _ = self.sess.run(fetches, feed_dict)

        n += 1

        if is_validation == False:
          self.train_writer.add_summary(summary, step)

        # reset sstart and ustart with zeros. Was it not already initialzed as zeros above??
        sstart = np.zeros_like(sstart, dtype=np.bool)
        ustart = np.zeros_like(ustart, dtype=np.bool)

        c.append(cost)

        if np.isnan(cost):
          logger.error('NaN error!')
          self.error_during_train = True
          return

      # move session start forward in time by the number of items session_minlen which were processed
      session_start = session_start + session_minlen - 1

      # identifieds sessions which have been finished processing
      session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
      # reset the session hidden state
      sstart[session_start_mask] = True

      # for sessions which are finished, move to next session
      for idx in session_start_mask:
        # go to next session by incrementing
        session_iters[idx] += 1
        # if there are no further sessions, end epoch
        if session_iters[idx] + 1 >= len(offset_sessions):
          finished = True
          break
        # else assign new sessions/begin of sessions
        session_start[idx] = offset_sessions[session_iters[idx]]
        session_end[idx] = offset_sessions[session_iters[idx] + 1]

      # identifies which user has been finished processing
      user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
      # reset the corresponding User hidden state at user change
      ustart[user_change_mask] = True

      # for users which are finished, move to next user
      for idx in user_change_mask:
        # go to next user by incrementing
        user_maxiter += 1
        # if there are no further users, end epoch
        if user_maxiter + 1 >= len(offset_users):
          finished = True
          break
        # fill batch (of fixed size) with new user
        user_iters[idx] = user_maxiter
        user_start[idx] = offset_users[user_maxiter]
        user_end[idx] = offset_users[user_maxiter + 1]
        # add new sessions for new user
        session_iters[idx] = user_indptr[user_maxiter]
        session_start[idx] = offset_sessions[session_iters[idx]]
        session_end[idx] = offset_sessions[session_iters[idx] + 1]

    # calculates and returns average cost/loss
    avgc = np.mean(c)

    return avgc

  def fit(self, train_data, valid_data=None, patience=3, margin=1.003):
    '''
    Training. Pre-process training data and perform iterations over epochs.
    :param train_data:
    :param valid_data:
    :param patience:
    :param margin:
    :return:
    '''

    self.error_during_train = False

    # map item ids ItemIdx to items
    itemids = train_data[self.item_key].unique()
    self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
    train_data = pd.merge(train_data,
                          pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                          on=self.item_key, how='inner')

    # get index pointers for users and sessions
    user_indptr, offset_sessions = self.preprocess_data(train_data)

    # if validation set exists, then repeat above for validation set
    user_indptr_valid, offset_sessions_valid = None, None
    if valid_data is not None:
      valid_data = pd.merge(valid_data,
                            pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                            on=self.item_key, how='inner')
      user_indptr_valid, offset_sessions_valid = self.preprocess_data(valid_data)

    #
    epoch = 0
    best_valid = None
    my_patience = patience

    # why two my_patience?
    while epoch < self.n_epochs and my_patience > 0 and my_patience > 0:

      # train on batches of epoch, return average cost/loss
      train_cost = self.iterate(train_data, offset_sessions, user_indptr)

      if np.isnan(train_cost):
        print('Epoch {}: Nan error!'.format(epoch, train_cost))
        return

      # check on validation data for estimating the loss/cost
      if valid_data is not None:
        valid_cost = self.iterate(valid_data, offset_sessions_valid, user_indptr_valid)
        if best_valid is None or valid_cost < best_valid:
          best_valid = valid_cost
          my_patience = patience
        elif valid_cost >= best_valid * margin:
          my_patience -= 1
        logger.info(
          'Epoch {} - train cost: {:.4f} - valid cost: {:.4f}'.format(epoch, train_cost, valid_cost, my_patience)
        )
      else:
        logger.info('Epoch {} -train cost: {:.4f}'.format(epoch, train_cost))

      epoch += 1

      # temporarily commented
      #self.saver.save(self.sess, '{}/hgru-model'.format(self.checkpoint_dir), global_step=epoch)
