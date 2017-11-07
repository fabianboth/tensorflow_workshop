import tensorflow as tf
from linear_regression_model import Regressor
# from fcn_regression_model import Regressor
from data_linear import data_generator_linear, batch_generator


def train(session, model, data_generator, training_steps, logdir='./logfiles/linear_reg'):
    """ This method contains the training loop and writes intermediate outputs to a summary
    
    :param session: The open sessino with the currently loaded default graph
    :param model: The current model. This class contains various tensors and etc for reference.
    :param data_generator: An generator object which iterates over all data
    :param training_steps: Number of training steps
    :param logdir: The folder in which summary logs are stored.
    """
    # create summary writer and write graph file

    # iterate over data and train
        # training finished?

        # create feed dict with tensors (data[:, :1],  data[:, 1:])

        # execute fetches with summary

        # write summary to file


def start_training(logdir, training_steps=1000, batch_size=10):
    """ This method prepares the training process and subsequently starts it.
    This involves creating a data iterator, a session and a respective model to solve the problem
    
    :param logdir: The folder in which summary logs are stored.
    :param training_steps: Number of training steps to perform
    :param batch_size: The number of samples in a mini-batch, i.e. samples per gradient update
    """
    # setup data generator and batch aggregation

    # create session
        # construct model and initialize

        # start train loop


if __name__ == '__main__':
    logdir = './logfiles/linear_reg_1'
    training_steps = 1000
    batch_size = 10

    start_training(training_steps=training_steps, batch_size=10, logdir=logdir)
