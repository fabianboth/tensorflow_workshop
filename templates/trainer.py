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
    summary_writer = tf.summary.FileWriter(logdir=logdir)
    summary_writer.add_graph(graph=tf.get_default_graph())  # write meta graph to logfile

    for n, data in enumerate(data_generator):
        if n % 100 == 0:
            print('Training step: %d' % n)

        # training finished?
        if n >= training_steps:
            break

        feed_dict = {model.x: data[:, :1], model.target: data[:, 1:]}
        _, summary = session.run(fetches=[model.train_op, model.summary_train],
                                 feed_dict=feed_dict)

        # write summary to file
        summary_writer.add_summary(summary=summary, global_step=n)


def start_training(logdir, training_steps=1000, batch_size=10):
    """ This method prepares the training process and subsequently starts it.
    This involves creating a data iterator, a session and a respective model to solve the problem
    
    :param logdir: The folder in which summary logs are stored.
    :param training_steps: Number of training steps to perform
    :param batch_size: The number of samples in a mini-batch, i.e. samples per gradient update
    """
    data_generator = data_generator_linear(num_points=200, sd=0.4)  # generates and iterates data points
    data_generator = batch_generator(data_generator=data_generator, batch_size=batch_size) # aggregates to batch

    with tf.Session() as sess:
        model = Regressor()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        train(logdir=logdir, session=sess, model=model,
              data_generator=data_generator, training_steps=training_steps)


if __name__ == '__main__':
    logdir = './logfiles/linear_reg_1'
    # logdir = './logfiles/fcn_reg_2'
    training_steps = 1000
    batch_size = 10

    start_training(training_steps=training_steps, batch_size=10, logdir=logdir)
