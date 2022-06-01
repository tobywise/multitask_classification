from email.generator import Generator
import os
import mne
from tqdm.auto import tqdm
import requests
import shutil
import numpy as np
import warnings
from typing import Tuple, List, Union
from tensorflow.data import Dataset
from tensorflow.image import per_image_standardization
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA


def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Scales X data so that each feature has zero mean and unit variance across samples.

    Args:
        X (np.ndarray): X data, can be 3D (trials, channels, time) or 4D (subjects, trials, channels, time)

    Returns:
        np.ndarray: Scaled X data
    """

    if X.ndim == 3:
        return (X - X.mean(axis=0)) / X.std(axis=0)
    elif X.ndim == 4:
        return (X - X.mean(axis=1)[:, None, ...]) / X.std(axis=1)[:, None, ...]


def shuffle_data(
    X: np.ndarray, y: np.ndarray, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffles X and Y data across trials

    Args:
        X (np.ndarray): X data, can be 3D (trials, channels, time) or 4D (subjects, trials, channels, time)
        y (np.ndarray): Y data, can be 3D (subjects, trials) or 1D (trials)
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Shuffled X and Y data
    """

    if X.ndim == 3:
        idx = np.arange(X.shape[0])
    else:
        idx = np.arange(X.shape[1])  # Shuffle trials

    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    if X.ndim == 3:
        return X[idx, ...], y[idx, ...]
    else:
        return X[:, idx, ...], y[:, idx, ...]


def load_MEG_dataset(
    subject_ids: List[str],
    mode: str = "individual",
    output_format: str = "numpy",
    trial_data_format: str = "2D",
    data_location: str = "./data/",
    center_timepoint: int = 20,
    window_width: List[int] = [-5, 6],
    shuffle: bool = False,
    pca_n_components: int = None,
    training: bool = True,
    train_test_split: float = 0.75,
    batch_size: int = 32,
    scale: bool = True,
    seed: int = 0,
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray]], Tuple[np.ndarray, np.ndarray], Dataset
]:
    """
    Downloads and loads MEG data from the specified subject ids. Data is returned as a generator of batches, if batch is True, or
    a generator with a single batch if batch is False.

    Data is returned as an X array representing the MEG data and a y array representing the labels.

    Args:
        subject_ids (list): List of subject ids to load as strings, e.g. (['001', '002']).
        mode (str, optional): How to organise the data, can be one of 'individual', 'concatenate', or 'stack'.
        If 'individual', a list of datasets is returned, each representing a single subject. If 'concatenate', all datasets are concatenated
        together and returned as a single dataset. If 'stack', all datasets are stacked together and returned as a single dataset.
        Defaults to 'individual'.
        output_format (str, optional): Format of the data, can be one of 'numpy' or 'tf'. If 'numpy', the data is returned as a numpy array. If
        'tf', data is returned as a tensorflow dataset. Defaults to 'numpy'.
        trial_data_format (str, optional): Format of the trial data, can be one of '2D' or '1D'. If '2D', the data for each trial is
        returned as a 2D array, where the first dimension is the channel number and the second dimension is the time index. If '1D', the data
        for each trial is returned as a 1D array, with channels and time points flattened. Defaults to '2D'.
        data_location (str, optional): Location to store the data. Defaults to "./data/".
        shuffle (bool, optional): If True, shuffles the data across trials. Defaults to False.
        pca_n_components (int, optional): Number of components to use when running PCA. If None, PCA is not performed. Defaults to None.
        training (bool, optional): If True, returns training data. Defaults to True.
        train_test_split (float, optional): Percentage of data to use for training. Defaults to 0.75.
        center_timepoint (int, optional): Index of the center time point for the classifier, post stimulus
        onset. Defaults to 20.
        window_width (List[int], optional): Window of data to use for the classifier. Defaults to [-5, 6].
        batch_size (int, optional): Batch size, to be used if using TF format. Defaults to 32.
        scale (bool, optiomal): Whether to scale the data. Defaults to True.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        Union[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[np.ndarray, np.ndarray], Dataset]: X and y data, or a generator of batches.
    """

    # Check subject ids are valid
    valid_ids = [str(i).zfill(3) for i in range(1, 29)]
    for sub in subject_ids:
        if not sub in valid_ids:
            raise ValueError(f"Invalid subject id: {sub}")

    mode = mode.lower()
    output_format = output_format.lower()

    # Check mode is valid
    if not mode in ["individual", "concatenate", "stack"]:
        raise ValueError(f"Invalid mode: {mode}")

    # Check output format is valid
    if not output_format in ["numpy", "tf"]:
        raise ValueError(f"Invalid output format: {output_format}")

    # Haven't implemented stacking for TF datasets yet # TODO
    if mode == "stack" and output_format == "tf":
        raise NotImplementedError("Stack mode not yet supported for TF datasets")

    n_stim = 14  # Number of stimuli

    X_append = []
    y_append = []

    for sub in subject_ids:
        print(f"Loading subject {sub}")

        if not isinstance(sub, str):
            raise TypeError("Subject ID must be a string, not a {}".format(type(sub)))

        # Create directory for subject if it doesn't exist
        if not os.path.exists(os.path.join(data_location, "sub-{0}".format(sub))):
            os.makedirs(os.path.join(data_location, "sub-{0}".format(sub)))

        # Download data if it doesn't exist
        if not os.path.exists(
            os.path.join(
                data_location,
                "sub-{0}".format(sub),
                "sub-{0}_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz".format(
                    sub
                ),
            )
        ):

            url = (
                "https://openneuro.org/crn/datasets/ds003682/snapshots/1.0.0/files/derivatives:"
                "preprocessing:sub-{0}:localiser:sub-{0}_ses-01_task-AversiveLearningReplay"
                "_run-localiser_proc_ICA-epo.fif.gz".format(sub)
            )

            # https://www.alpharithms.com/progress-bars-for-python-downloads-580122/
            print("Downloading data for subject {}".format(sub))
            with requests.get(
                url,
                stream=True,
                headers={"Accept-Encoding": None, "Content-Encoding": "gzip"},
            ) as r:
                with open(
                    os.path.join(
                        data_location,
                        "sub-{0}".format(sub),
                        "sub-{0}_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz".format(
                            sub
                        ),
                    ),
                    "wb",
                ) as f:
                    shutil.copyfileobj(r.raw, f)
            print("Download complete")

        localiser_epochs = mne.read_epochs(
            os.path.join(
                data_location,
                "sub-{0}".format(sub),
                "sub-{0}_ses-01_task-AversiveLearningReplay_run-localiser_proc_ICA-epo.fif.gz".format(
                    sub
                ),
            ),
            verbose="ERROR",
        )

        print('Data loaded')

        # Get epoch data
        X_raw = (
            localiser_epochs.get_data()
        )  # MEG signals: n_epochs, n_channels, n_times (exclude non MEG channels)
        y_raw = localiser_epochs.events[:, 2]  # Get event types

        # select events and time period of interest
        picks_meg = mne.pick_types(localiser_epochs.info, meg=True, ref_meg=False)
        event_selector = y_raw < n_stim * 2 + 1
        X_raw = X_raw[event_selector, ...]
        y_raw = y_raw[event_selector]
        X_raw = X_raw[:, picks_meg, :]

        # Run PCA
        if pca_n_components is not None:
            print("Running PCA")
            pca = UnsupervisedSpatialFilter(PCA(pca_n_components), average=False)
            X_raw = pca.fit_transform(X_raw)
            print("PCA complete")

        assert (
            len(np.unique(y_raw)) == n_stim
        ), "Found {0} stimuli, expected {1}".format(len(np.unique(y_raw)), n_stim)

        # Samples before stimulus onset
        prestim_samples = int(
            np.abs(localiser_epochs.tmin * localiser_epochs.info["sfreq"])
        )

        # Actual index of the center of the clas~sification window, accounting for the prestimulus period
        classifier_center_idx = prestim_samples + center_timepoint

        # Get data
        X, y = (X_raw.copy(), y_raw.copy())
        X = X[
            ...,
            classifier_center_idx
            + window_width[0] : classifier_center_idx
            + window_width[1],
        ]


        if scale:
            for i in range(X.shape[0]):
                a = X[i, ...].std()
                b = 1 / np.sqrt(X[i, ...].size)
                c = a / b
                X[i, ...] = (X[i, ...] - X[i, ...].mean())  / np.max(X[i, ...].std(), )

        # Check number of trials
        if X.shape[0] != 900:
            warnings.warn(
                "Found {0} trials for subject {1}, expected {2} - skipping".format(
                    X.shape[0], sub, 900
                )
            )

        else:
            X_append.append(X)
            y_append.append(y)

        print("Subject {0} complete".format(sub))
        print('--------------------------------------')



    # Concatenate/stack data data
    if len(X_append) > 1:
        if mode in [
            "individual",
            "stack",
        ]:  # Stack for now even if indivual, then split later
            X = np.stack(X_append)
            y = np.stack(y_append)
        elif mode == "concatenate":
            X = np.concatenate(X_append, axis=0)
            y = np.concatenate(y_append, axis=0)

    # Get data for single subject
    if len(X_append) == 0:
        if mode == "stack":
            X = X[None, ...]
            y = y[None, ...]
        if mode == "concatenate":
            X = X_append[0]
            y = y_append[0]

    # Train test split - TODO this splits only across trials currently
    if training:
        if X.ndim == 3:
            X = X[: int(X.shape[0] * train_test_split), ...]
            y = y[: int(y.shape[0] * train_test_split), ...]
        else:
            X = X[:, : int(X.shape[1] * train_test_split), ...]
            y = y[:, : int(y.shape[1] * train_test_split), ...]
    else:
        if X.ndim == 3:
            X = X[int(X.shape[0] * train_test_split) :, ...]
            y = y[int(y.shape[0] * train_test_split) :, ...]
        else:
            X = X[:, int(X.shape[1] * train_test_split) :, ...]
            y = y[:, int(y.shape[1] * train_test_split) :, ...]

    # Scale
    # if scale:
    #     X = scale_data(X)

    # Reshape data if necessary
    if trial_data_format == "1D":
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim == 4:
            X = X.reshape(X.shape[0], X.shape[1], -1)

    # Shuffle numpy format data if needed
    if output_format == "numpy":
        if shuffle:
            X, y = shuffle_data(X, y, seed=seed)
        if mode == "individual":
            X = np.split(X, X.shape[0], axis=0)
            y = np.split(y, y.shape[0], axis=0)
            X = [i.squeeze() for i in X]
            y = [i.squeeze() for i in y]

        return (X, y)

    # Convert to TF dataset, shuffle and batch
    else:
        if mode == "individual":
            ds = [
                Dataset.from_tensor_slices({"image": X, "label": y}).cache().repeat(None)
                for X, y in zip(X, y)
            ]
            if shuffle:
                ds = [i.shuffle(10 * batch_size, seed=seed) for i in ds]
            ds = [i.batch(batch_size).as_numpy_iterator() for i in ds]
        else:
            ds = Dataset.from_tensor_slices({"image": X, "label": y}).cache().repeat(None)
            if shuffle:
                ds = ds.shuffle(10 * batch_size, seed=0)
            ds = ds.batch(batch_size).as_numpy_iterator()

        return ds
