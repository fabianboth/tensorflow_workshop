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
        with tf.variable_scope('LinearRegression'):
            # define placeholders


            # ---- construct regression relation
            # Variable a is the first (and only) coefficient

            # Variable b is the bias of the regression


            # store internal variables (alternatively also an internal dict can be used)

    def construct_optimizer(self, y, target, learning_rate=0.001):
        """ Here a loss and an optimizer is defined for solving the problem
        
        :param y: The logits (prediction) of the model
        :param target: The ground truth value
        :param learning_rate: This defines the update step size
        """
        with tf.variable_scope('optimizer'):
            # squared error


            # store internal variables

    def add_summaries(self):
        """ In this method summaries for the loss and the regression variables a and b are created
        """
        with tf.variable_scope('summaries'):
            # create scalar summaries

            # fuse all summaries in a single operation

            # store internal variables
