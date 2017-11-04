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
        with tf.variable_scope('Regression'):
            # define placeholders
            x = tf.placeholder(dtype=tf.float32, shape=[None, ], name='x_input')
            target = tf.placeholder(dtype=tf.float32, shape=[None, ], name='target')

            # ---- construct regression relation
            # Variable a is the first (and only) coefficient
            a = tf.get_variable(name='a', shape=[1], initializer=tf.constant_initializer)
            # Variable b is the bias
            b = tf.get_variable(name='b', shape=[1], initializer=tf.constant_initializer)

            y = tf.multiply(x, a) + b  # basic operators are also available (+, *, ...)

            # store internal variables (alternatively also an internal dict can be used)
            self.x = x
            self.target = target
            self.a = a
            self.b = b
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
            summary.append(tf.summary.scalar(tensor=tf.reduce_mean(self.a), name='a'))
            summary.append(tf.summary.scalar(tensor=tf.reduce_mean(self.b), name='b'))

            # fuse all summaries in a single operation
            summary_train = tf.summary.merge(inputs=summary)

            # store internal variables
            self.summary_train = summary_train
