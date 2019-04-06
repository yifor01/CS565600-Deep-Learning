# import the required libraries
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=8,suppress=True)
import os

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import tensorflow as tf
class SRNN_Model():
  
    def __init__(self, hps, model_name='sketch_rnn'):
        self.hps = hps
        self.model_name=model_name
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            self.build_model(hps)
    
    def build_model(self, hps):
      
        # input and output
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_sequence = tf.placeholder(dtype=tf.float32, shape=[None, hps.max_seq_len+1, 5], name='input_sequence')
        input_sequence = self.input_sequence[:, :-1]
        output_sequence = self.input_sequence[:, 1:, :]
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(hps.dec_rnn_size)
        self.zero_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        output, _ = tf.nn.dynamic_rnn(self.lstm_cell, input_sequence, initial_state=self.zero_state)
        output = tf.reshape(output, [-1, hps.dec_rnn_size])
        
        # for each input timestamp, output parameters for mixture of gaussian
        num_hidden = hps.num_mixture * 6 + 3 
        def feed_forward(output):
            with tf.variable_scope('feed_forward', reuse=tf.AUTO_REUSE):
                return tf.layers.dense(output, num_hidden)
        output = feed_forward(output)
        ##############################################################################
        
        """
        Loss for mixture of multivariate gaussian. Given (prev_delta_x, prev_delta_y, prev_p1, prev_p2, prev_p3), 
        we want the mixture to output high probility P(delta_x, delta_y|prev_delta_x, prev_delta_y, prev_p1, prev_p2, prev_p3),
        see equation (9) at https://arxiv.org/pdf/1704.03477.pdf
        
        parameters:
            pi: weight of each mixture, shape (batch_size*max_seq_len, num_mixture)
            mu1, mu2: mu of (delta_x, delta_y), shape (batch_size*max_seq_len, num_mixture)
            sigma1, sigma2: sigma of (delta_x, delta_y), shape (batch_size*max_seq_len, num_mixture)
            corr: correction of (delta_x, delta_y), shape (batch_size*max_seq_len, num_mixture)
        """
        output_params = tf.split(output, num_or_size_splits=[3]+[hps.num_mixture]*6, axis=1)
        output_pen_logits, output_mixture_pi, output_mu1, output_mu2, output_sigma1, output_sigma2, output_corr = output_params
        
        # softmax all the pi's and pen states:
        output_mixture_pi = tf.nn.softmax(output_mixture_pi)
        output_pen_pi = tf.nn.softmax(output_pen_logits)
        
        # exponentiate the sigmas and also make corr between -1 and 1.
        output_sigma1 = tf.exp(output_sigma1)
        output_sigma2 = tf.exp(output_sigma2)
        output_corr = tf.tanh(output_corr)
        
        x1 = tf.reshape(output_sequence[:, :, 0], [-1, 1])
        x2 = tf.reshape(output_sequence[:, :, 1], [-1, 1])
        
        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            """
            Returns P(delta_x, delta_y|prev_delta_x, prev_delta_y, prev_p1, prev_p2, prev_p3), 
            see equation (24) of http://arxiv.org/abs/1308.0850 or 
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
            """
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            # eq 25
            z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
                 2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
            neg_rho = 1 - tf.square(rho) + 1e-6 # avoid divide by zero
            result = tf.exp(tf.div(-z, 2 * neg_rho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
            result = tf.div(result, denom)
            self.denom = denom
            self.neg_rho = neg_rho
            self.s1s2 = s1s2
            self.s1 = s1
            self.s2 = s2
            return result

        point_prob = tf_2d_normal(x1, x2, output_mu1, output_mu2, output_sigma1, output_sigma2, output_corr)
        point_prob = tf.multiply(point_prob, output_mixture_pi) # multiply weight of each mixture
        point_prob = tf.reduce_sum(point_prob, axis=1)
        
        # loss for indicating if pen should stop
        pen_labels = tf.reshape(output_sequence[:, :, 2:], [-1, 3]) # (batch_size*max_seq_len ,3)
        pen_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pen_labels, logits=output_pen_logits)
        self.pen_loss = tf.reduce_mean(pen_loss)
        
        # loss for delta x,y
        delta_xy_loss = -tf.log(point_prob + 1e-6)  # avoid log(0)
        mask = 1.0 - pen_labels[:, 2]    
        mask = tf.reshape(mask, [-1, 1])        
        delta_xy_loss = delta_xy_loss * mask
        self.delta_xy_loss = tf.reduce_mean(delta_xy_loss)
        
        
        self.loss = self.pen_loss + self.delta_xy_loss

        # optimize rnn
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.learning_rate = tf.get_variable(name='learning_rate', initializer=tf.constant(hps.learning_rate), trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/')
        grad_vars = optimizer.compute_gradients(self.loss, var_list=var_list)
        grad_vars = [(tf.clip_by_value(grad, -hps.grad_clip, hps.grad_clip), var) for grad, var in grad_vars]
        self.train_op = optimizer.apply_gradients(grad_vars, global_step=self.global_step, name='train_step')
        
        #################### tensor for generating a point ############################
        self.prev_state = self.lstm_cell.zero_state(1, dtype=tf.float32)
        self.prev_point = tf.placeholder(tf.float32, shape=[1,5], name='prev_point')
        output, self.next_state = self.lstm_cell(self.prev_point, self.prev_state)
        output = tf.reshape(output, [-1, hps.dec_rnn_size])
        output = feed_forward(output)
        self.output_pi, self.output_mu1, self.output_mu2, self.output_sigma1, self.output_sigma2, self.output_corr = \
                                                              tf.split(output[:,3:], num_or_size_splits=6, axis=1)
        pen_logits = output[:,:3]
        self.output_pi = tf.nn.softmax(self.output_pi)
        self.pen_pi = tf.nn.softmax(pen_logits)
        self.output_sigma1 = tf.exp(self.output_sigma1)
        self.output_sigma2 = tf.exp(self.output_sigma2)
        self.output_corr = tf.tanh(self.output_corr)
        
    def generate_stroke(self, sess, prev_sketch, temperature = 0.1, greedy = False):
        """
        *****************************IMPORTANT*****************************
        1. your model must have this function, or error happen when evaluation.
        2. the length of stroke must less than 10, which means you can at most generate 10 points.
        *******************************************************************
        this function return a stroke given previous generated sketch
        """
        # feed previous sketch to get hidden state
        prev_state = sess.run(self.zero_state, feed_dict={self.batch_size: 1})
        for i in range(len(prev_sketch)-1):
            feed_dict = {
                    self.prev_state: prev_state,
                    self.prev_point: [prev_sketch[i]],
            }
            prev_state = sess.run(self.next_state, feed_dict)
        
        # start to generate next stroke
        prev_point = [prev_sketch[-1]]
        generated_stroke = []
        while len(generated_stroke) < 10: 
            feed_dict = {
                self.prev_state: prev_state,
                self.prev_point: prev_point,
            }
            params = sess.run([
                self.output_pi, self.output_mu1, self.output_mu2, self.output_sigma1, self.output_sigma2, self.output_corr,
                self.pen_pi, self.next_state
            ], feed_dict)

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params
            
            # sample index of bivarite normal in mixture to use
            idx = get_pi_idx(np.random.random(), o_pi[0], temperature, greedy)
            
            # sample index of pen state
            idx_eos = get_pi_idx(np.random.random(), o_pen[0], temperature, greedy)
            eos = [0, 0, 0]
            eos[idx_eos] = 1

            # use one bivarite normal to generate next (delta_x, delta_y)
            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                                  o_sigma1[0][idx], o_sigma2[0][idx],
                                                  o_corr[0][idx], np.sqrt(temperature), greedy)

            prev_point = np.zeros([1,5])
            prev_point[0, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
            prev_state = next_state
            generated_stroke.append(prev_point)
            
            # if this stroke stop
            if prev_point[0][4]==1:
                generated_stroke[-1][0][:2] = 0
            if prev_point[0][4]==1 or prev_point[0][3]==1:
                break
        # pen state should end with p2=1 or p3=1
        if generated_stroke[-1][0][2] == 1:
            generated_stroke[-1][0][2:] = [0.,1.,0.]
        return np.concatenate(generated_stroke, axis=0)    
  
    def generate_sketches(self, sess, num_generate, temperature, greedy=False):
        initial_point = np.array([[0.,0.,1.,0.,0.]])
        initial_state = sess.run(self.zero_state, feed_dict={self.batch_size: 1})
        return_sketches = []
        for i in range(num_generate):
            sketch = [initial_point]
            prev_point = initial_point
            prev_state = initial_state
            for j in range(self.hps.max_seq_len):
                feed_dict = {
                    self.prev_state: prev_state,
                    self.prev_point: prev_point,
                }
                params = sess.run([
                    self.output_pi, self.output_mu1, self.output_mu2, self.output_sigma1, self.output_sigma2, self.output_corr,
                    self.pen_pi, self.next_state
                ], feed_dict)

                [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params
                idx = get_pi_idx(random.random(), o_pi[0], temperature, greedy)

                idx_eos = get_pi_idx(random.random(), o_pen[0], temperature, greedy)
                eos = [0, 0, 0]
                eos[idx_eos] = 1

                next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                                      o_sigma1[0][idx], o_sigma2[0][idx],
                                                      o_corr[0][idx], np.sqrt(temperature), greedy)

                prev_point = np.zeros([1,5])
                prev_point[0, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]
                prev_state = next_state
                
                # select a multivariate normal in mixture to draw next point
                sketch.append(prev_point)
                if prev_point[0][4]==1:
                    sketch[-1][0][:2] = 0
                    break
            return_sketches.append(np.concatenate(sketch, axis=0))
        return return_sketches
  
    def train(self, sess, x_train, x_valid, x_test, num_epoch=100, batch_size=100, patience = 20):
        start = time.time()
        best_valid_cost = np.inf
        counter = 0
        for epoch in range(num_epoch):
            def gen_batch(x):
                shuffle_idx = np.random.permutation(len(x))
                x = x[shuffle_idx]
                num_batch = len(x)//batch_size
                for i in range(num_batch):
                    yield x[i*batch_size:(i+1)*batch_size]
            losses = []
            delta_xy_losses = []
            pen_losses = []
            for x_batch in gen_batch(x_train):
                step = sess.run(self.global_step)
                curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                                      (hps.decay_rate)**step + hps.min_learning_rate)    
                feed_dict = {
                    self.input_sequence: x_batch,
                    self.learning_rate: curr_learning_rate,
                    self.batch_size: len(x_batch)
                }
                (loss, delta_xy_loss, pen_loss, _) = sess.run([self.loss, self.delta_xy_loss, 
                                                               self.pen_loss, self.train_op], feed_dict)
                losses.append(loss)
                delta_xy_losses.append(delta_xy_loss)
                pen_losses.append(pen_loss)
                if step % 20 == 0 and step > 0:
                    end = time.time()
                    time_taken = end - start
                    tf.logging.info(('step: {}, learning_rate: {:.4f}, loss: {:.4f}, xy_loss: {:.4f}, '+
                                     'pen_loss: {:.4f}, time_taken: {:.2f}').format(step, curr_learning_rate, 
                                      np.mean(losses), np.mean(delta_xy_losses), np.mean(pen_losses), time_taken))
                    losses = []
                    delta_xy_losses = []
                    pen_losses = []
                    start = time.time()
                if step % 1000 == 0 and step > 0:
                    N = 10
                    sketches = self.generate_sketches(sess, num_generate=N, temperature=0.5)
                    reconstructions = []
                    for i in range(N):
                        reconstructions.append([to_normal_strokes(sketches[i]), [0, i]])
                    stroke_grid = make_grid_svg(reconstructions)
                    draw_strokes(stroke_grid)

                if step % self.hps.save_every == 0 and step > 0:
                    start = time.time()
                    test_losses = []
                    test_delta_xy_losses = []
                    test_pen_losses = []
                    for x_batch in gen_batch(x_valid):
                        feed_dict = {
                            self.input_sequence: x_batch,
                            self.batch_size: len(x_batch)
                        }

                        (loss, delta_xy_loss, pen_loss) = sess.run([self.loss, self.delta_xy_loss, self.pen_loss], feed_dict)
                        test_losses.append(loss)
                        test_delta_xy_losses.append(delta_xy_loss)
                        test_pen_losses.append(pen_loss)

                    end = time.time()
                    time_taken = end - start
                    tf.logging.info(('validation, step: {}, loss: {:.4f}, xy_loss: {:.4f}, '+
                                     'pen_loss: {:.4f}, time_taken: {:.2f}').format(step, np.mean(test_losses), 
                                      np.mean(test_delta_xy_losses), np.mean(test_pen_losses), time_taken))                    
                    start = time.time()

                    if np.mean(test_losses) < best_valid_cost:
                        best_valid_cost = np.mean(test_losses)
                        #self.save_model(sess, step=step)
                        self.save_model(sess)
                        test_losses = []
                        test_delta_xy_losses = []
                        test_pen_losses = []
                        for x_batch in gen_batch(x_test):
                            feed_dict = {
                                self.input_sequence: x_batch,
                                self.batch_size: len(x_batch)
                            }

                            (loss, delta_xy_loss, pen_loss) = sess.run([self.loss, self.delta_xy_loss, self.pen_loss], feed_dict)
                            test_losses.append(loss)
                            test_delta_xy_losses.append(delta_xy_loss)
                            test_pen_losses.append(pen_loss)

                        end = time.time()
                        time_taken = end - start
                        tf.logging.info(('testing, step: {}, loss: {:.4f}, xy_loss: {:.4f}, '+
                                         'pen_loss: {:.4f}, time_taken: {:.2f}').format(step, np.mean(test_losses), 
                                          np.mean(test_delta_xy_losses), np.mean(test_pen_losses), time_taken))
                        counter = 0
                    else:
                        counter += 1
                        if counter > patience:
                            tf.logging.info('early stop!!')
                            return
                          
    def save_model(self, sess, checkpoint_dir='./checkpoints_v2',step=None):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/'))
        if step is not None:
            saver.save(sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
            tf.logging.info('model save to {}-{}'.format(os.path.join(checkpoint_dir, self.model_name), step))
        else:
            saver.save(sess, os.path.join(checkpoint_dir, self.model_name))
            tf.logging.info('model save to {}'.format(os.path.join(checkpoint_dir, self.model_name)))
        
    def load_model(self, sess, checkpoint_dir='./checkpoints', step=None):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name+'/'))
        if step is not None:
            saver.restore(sess, os.path.join(checkpoint_dir, self.model_name+'-{}'.format(step)))
        else:
            saver.restore(sess, os.path.join(checkpoint_dir, self.model_name))
            

def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      data_set=['aaron_sheep.npz'],  # Our dataset.
      save_every=2000,  # Number of batches per checkpoint creation.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=1024,  # Size of decoder.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.0001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True  # Is model training? Recommend keeping true.
  )
  return hparams
# function used to generate next point
def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
        return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]
