import numpy as np
from matplotlib import pyplot as plt


def data_linear(num_points=200, sd=0.4):
    """ Generates points on the quadrant diagonal which are noised with a gaussian signal.
    The generated points lie within [0, 10] and have a linear relationship.
    
    :param num_points: Number of points to generate
    :param sd: Standard deviation of the gaussian noise
    :return: 2D numpy array with x coordinates in the first col and y in the second col.
    """
    # equal spaced points
    points = np.linspace(start=0, stop=10, num=num_points)

    x = points + np.random.normal(loc=0, scale=sd, size=num_points)
    y = points + np.random.normal(loc=0, scale=sd, size=num_points)

    coordinates = np.array([x, y]).T

    return coordinates


def data_generator_linear(num_points=200, sd=0.4):
    """ Wrapper to create an infinite data point iterator over a fixed set of points
        
    :param num_points: Number of points which will be created
    :param sd: Standard deviation of the (linearly correlated) points
    :return: Yields single data points (shuffled ordering)
    """
    data = data_linear(num_points=num_points, sd=sd)
    np.random.shuffle(data)

    while True:
        # iterate over all data points
        for d in data:
            yield d

        # shuffle data before next round
        np.random.shuffle(data)


def batch_generator(data_generator, batch_size):
    """ Wrapper which aggregates single data points to batches of data points and yields them.
    
    :param data_generator: A generator object which is iterable and yields data points
    :param batch_size: Number of data samples in a single batch
    :return: 2D np array with data points in rows and [x, y] in the columns.
    """
    all_data = list()
    for data in data_generator:
        # batch has been filled
        if len(all_data) >= batch_size:
            yield np.stack(all_data, axis=0)
            all_data = list()  # reset data list

        all_data.append(data)


# Testing the output
if __name__ == '__main__':
    points = data_linear()
    np.random.shuffle(points)

    plt.scatter(x=points[:, 0], y=points[:, 1])
    plt.show()
