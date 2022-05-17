import pytest
from sklearn.model_selection import train_test_split
from load_data import *
from tensorflow.data import Dataset

def test_data_mode_correct_concatenate():

    X, y = load_MEG_dataset(['001', '002'], mode='concatenate', output_format='numpy', trial_data_format='2D')
    assert X.shape == (675 * 2, 272, 11)
    assert y.shape == (675 * 2, )

def test_data_mode_correct_stack():
    X, y = load_MEG_dataset(['001', '002'], mode='stack', output_format='numpy', trial_data_format='2D')
    assert X.shape == (2, 675, 272, 11)
    assert y.shape == (2, 675)

def test_data_mode_correct_individual():
    X, y = load_MEG_dataset(['001', '002'], mode='individual', output_format='numpy', trial_data_format='2D')

    assert len(X) == 2
    assert len(y) == 2

    assert all([i.shape == (675, 272, 11) for i in X])
    assert all([i.shape == (675, ) for i in y])

def test_train_test_split_test():

    X, y = load_MEG_dataset(['001', '002'], mode='concatenate', output_format='numpy', trial_data_format='2D', training=False)
    assert X.shape == (int(1800 * .25), 272, 11)
    assert y.shape == (int(1800 * .25), )

def train_test_split_different_train_test_sizes():

    X, y = load_MEG_dataset(['001', '002'], mode='concatenate', output_format='numpy', trial_data_format='2D', train_test_split=.6)
    assert X.shape == (int(1800 * .6), 272, 11)
    assert y.shape == (int(1800 * .6), )

def test_data_format_correct_numpy():

    X, y = load_MEG_dataset(['001', '002'], mode='concatenate', output_format='numpy', trial_data_format='2D')
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_data_format_correct_tf():

    ds = load_MEG_dataset(['001', '002'], mode='concatenate', output_format='tf', trial_data_format='2D')
    assert isinstance(ds, Dataset)

def test_trial_data_shape_2D():

    X, y = load_MEG_dataset(['001', '002'], mode='stack', output_format='numpy', trial_data_format='2D')
    assert X.shape == (2, 675, 272, 11)

def test_trial_data_shape_1D():

    X, y = load_MEG_dataset(['001', '002'], mode='stack', output_format='numpy', trial_data_format='1D')
    assert X.shape == (2, 675, 272 * 11)

