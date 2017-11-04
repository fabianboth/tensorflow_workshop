import tensorflow as tf


class Regressor(object):
    """ A class that implements linear regression as tensorflow graph. """
    def __init__(self):
        self.build_graph()

        # additionally construct an optimizer and summaries
        self.construct_optimizer(y=self.y, target=self.target)
        self.add_summaries()

    def build_graph(self):
        """ In this method everything within up to the output logits is constructed. """
        with tf.variable_scope('FcnRegression'):
            # define placeholders
            x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_input')
            target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target')

            # ---- construct regression relation
            # First layer with 1 input and 10 outputs (10 neurons)
            w1 = tf.get_variable(name='w1', shape=[1, 10], initializer=tf.truncated_normal_initializer)
            b1 = tf.get_variable(name='b1', shape=[1], initializer=tf.constant_initializer)
            layer = tf.matmul(x, w1) + b1
            layer = tf.nn.elu(layer, name='layer_1')

            # Second layer with 10 inputs and 1 output (1 neuron)
            w2 = tf.get_variable(name='w2', shape=[10, 1], initializer=tf.truncated_normal_initializer)
            b2 = tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer)
            y = tf.matmul(layer, w2) + b2  # output y

            # store internal variables (alternatively also an internal dict can be used)
            self.x = x
            self.target = target
            self.y = y

    def construct_optimizer(self, y, target, learning_rate=0.001):
        """ Here a loss and an optimizer is defined for solving the problem
        
        :param y: The logits (prediction) of the model
        :param target: The ground truth value
        :param learning_rate: This defines the update step size
        """
        with tf.variable_scope('optimizer'):
            # squared error
            loss = tf.pow(tf.subtract(y, target), 2)
            loss = tf.reduce_mean(loss, name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss)

            # store internal variables
            self.loss = loss
            self.train_op = train_op

    def add_summaries(self):
        """ In this method summaries for the loss and the regression variables a and b are created
        """
        with tf.variable_scope('summaries'):
            summary = list()

            summary.append(tf.summary.scalar(tensor=self.loss, name='loss'))

            # fuse all summaries in a single operation
            summary_train = tf.summary.merge(inputs=summary)

            # store internal variables
            self.summary_train = summary_train
