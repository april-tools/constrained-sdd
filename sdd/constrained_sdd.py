import math
import pickle
from typing import Literal, Union
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

OBSTACLES_CLASSES = ["Building", "Obstacle"]
OFFROAD_CLASSES = ["Offroad"]


class PolytopeH:
    """
    Represents a polytope in H-representation, defined by a set of linear inequalities.

    Attributes:
        A (np.ndarray): A 2D numpy array representing the coefficients of the linear inequalities.
        b (np.ndarray): A 1D numpy array representing the constants of the linear inequalities.

    Args:
        A (np.ndarray): The coefficient matrix of shape (m, n), where m is the number of inequalities
                        and n is the dimensionality of the space.
        b (np.ndarray): The constant vector of shape (m,), corresponding to the right-hand side of the inequalities.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def rescale(self, scale: float):
        """
        Rescales the polytope by a given factor.

        Args:
            scale (float): The scaling factor.
        """
        # such that A * x = A' * (x * scale)
        # A' = A / scale

        self.A = self.A / scale


class DNF:
    """
    Represents a Disjunctive Normal Form (DNF) consisting of a list of polytopes.

    Attributes:
        polytopes (list[PolytopeH]): A list of PolytopeH objects that define the DNF.
    """

    def __init__(self, polytopes: list[PolytopeH]):
        self.polytopes = polytopes


class PolytopeV:
    """
    A class representing a polygon defined by its vertices.

    Attributes:
        vertices (np.ndarray): A NumPy array containing the vertices of the polygon.
                                Each vertex is typically represented as a coordinate pair (x, y).
    """

    def __init__(self, vertices: np.ndarray):
        self.vertices = vertices

    def rescale(self, scale: float):
        """
        Rescales the polygon by a given factor.

        Args:
            scale (float): The scaling factor.
        """

        self.vertices = self.vertices * scale


################# helpers ####

def filter_moving_trajectories(
    trajectories: dict[str, np.ndarray],
    threshold_variance: float = 20,
    threshold_speed: float = 0.1,
    speed_window: int = 10,
) -> dict[str, np.ndarray]:
    """
    Returns the trajectories that are moving, so trajectories that travel from its mean
    and with elements that have a speed greater than a threshold.
    """

    def to_non_stationary_trajectory(t: np.ndarray, threshold, windows_size=10):
        # calculate speed
        speed = np.linalg.norm(np.diff(t, axis=0), axis=1)
        # pad at the start to keep the same size
        # reuse the first value
        speed = np.pad(speed, (1, 0), mode="constant", constant_values=speed[0])
        # smooth it
        speed_smooth = np.convolve(
            speed, np.ones(windows_size) / windows_size, mode="same"
        )

        moving = speed_smooth > threshold
        t_non_stationary = t[moving, :]
        # print(t_non_stationary.shape)
        return t_non_stationary

    all_moving = {
        t_id: t
        for t_id, t in trajectories.items()
        if np.var(t, axis=0).sum() > threshold_variance
    }

    all_moving_non_stationary = {
        t_id: to_non_stationary_trajectory(t, threshold_speed, speed_window)
        for t_id, t in all_moving.items()
    }

    return all_moving_non_stationary


################# trajectory prediction ####


def single_trajectory_to_dataset_horizon_non_sampled(
    t_id: str, trajectory: np.ndarray, window_size: int, sampling_rate_window: int
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, int, int]]]:
    X = []
    y = []
    metadata: list[tuple[str, int, int]] = []
    real_window_size = window_size * sampling_rate_window
    real_one_step = 1 * sampling_rate_window

    for i in range(
        0, trajectory.shape[0] - max(real_window_size + real_one_step - 1, 0)
    ):
        history = trajectory[i : (i + real_window_size) : sampling_rate_window]
        assert history.shape[0] == window_size

        future_slice = trajectory[(i + (window_size - 1) * sampling_rate_window) :]
        X.append(history)
        y.append(future_slice)
        metadata.append((t_id, i, 0))  # type: ignore

    return np.stack(X, axis=0), y, metadata


def trajectories_to_dataset_horizon(
    trajectories: list[tuple[str, np.ndarray]],
    window_size: int,
    sampling_rate_window: int,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, int, int]]]:
    X = []
    y = []
    metadata: list[tuple[str, int, int]] = []
    for trajectory in trajectories:
        t_id, traj = trajectory
        X_traj, y_traj, metadata_traj = (
            single_trajectory_to_dataset_horizon_non_sampled(
                t_id, traj, window_size, sampling_rate_window
            )
        )
        X.append(X_traj)
        y.extend(y_traj)
        metadata.extend(metadata_traj)
    return np.concatenate(X, axis=0), y, metadata


def calc_rescale(h, w, target):
    if h > w:
        return target / h
    else:
        return target / w


class SampledHorizonDataset(Dataset):
    """
    Dataset for trajectory prediction with sampled horizon (y) during training.
    The dataset is sampled from the horizon (y) with a given distribution.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: list[np.ndarray],
        distribution: Literal["linear", "uniform", "mixture_uniform"] = "linear",
        bin_size_mixture: None | int = None,
        return_index=False,
    ):
        self.X = X
        self.y = y
        self.distribution = distribution
        self.bin_size_mixture = bin_size_mixture
        self.return_index = return_index

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x: np.ndarray = self.X[idx]
        y: np.ndarray = self.y[idx]
        len_y = y.shape[0]
        if self.distribution == "linear":
            # sample from y
            # with a linear decrasing density
            # so density is: 2/(len(y) ** 2) * (len(y) - x)

            # via inverse transform sampling
            u = np.random.uniform(0, 1)
            # inverse of cdf is len(y) - sqrt(1 - u) * len(y)
            continous_sample_idx = len_y - math.sqrt(1 - u) * len_y
            assert continous_sample_idx >= 0 and continous_sample_idx <= len_y
            sample_idx = math.floor(continous_sample_idx)
        elif self.distribution == "uniform":
            sample_idx = np.random.randint(0, len_y)
        elif self.distribution == "mixture_uniform":
            assert self.bin_size_mixture is not None
            sample_idx_uniform_length = np.random.randint(0, len_y)
            sample_idx_uniform_first_bin = np.random.randint(
                0, min(self.bin_size_mixture, len_y)
            )
            # choose between the two with equal probability
            if np.random.uniform(0, 1) < 0.5:
                sample_idx = sample_idx_uniform_length
            else:
                sample_idx = sample_idx_uniform_first_bin
        else:
            raise ValueError("Unknown distribution")
        if not self.return_index:
            return x, y[sample_idx]
        else:
            return x, y[sample_idx], sample_idx


def load_sdd_trajectories_from_file(
    file_path: str, sampled_horizon_kwargs: dict = {}
) -> tuple[SampledHorizonDataset, Dataset, Dataset]:
    with open(file_path, "rb") as f:
        data_trajectories = pickle.load(f)
    train_data = data_trajectories["train"]
    val_data = data_trajectories["val"]
    test_data = data_trajectories["test"]

    train_dataset = SampledHorizonDataset(
        train_data[0],
        train_data[1],
        **sampled_horizon_kwargs,
    )

    val_dataset = TensorDataset(torch.tensor(val_data[0]), torch.tensor(val_data[1]))
    test_dataset = TensorDataset(torch.tensor(test_data[0]), torch.tensor(test_data[1]))
    return train_dataset, val_dataset, test_dataset


################# dataset ####

def download_sdd_data(folder: str = "data/sdd"):
    # download github release
    import requests
    import zipfile
    import io

    url = (
        "https://github.com/april-tools/constrained-sdd/releases/download/data/sdd.zip"
    )
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder)
    # move files from sdd folder to folder
    import shutil

    for f in os.listdir(f"{folder}/sdd"):
        shutil.move(f"{folder}/sdd/{f}", f"{folder}/{f}")
    os.rmdir(f"{folder}/sdd")


class ConstrainedStanfordDroneDataset:
    def __init__(
        self,
        img_id: int,
        constraint_classes: list[str] = OBSTACLES_CLASSES + OFFROAD_CLASSES,
        sdd_data_path: str = "data/sdd",
        dequantized: bool = True,
        filter_moving: bool = True,
        download=True,
        rescale_coordinates: Union[
            bool, int
        ] = True,  # rescales the longer axis to 10 (or the value if integer)
    ):
        self.img_id = img_id
        self.constraint_classes = constraint_classes
        self.sdd_data_path = sdd_data_path
        self.filter_moving = filter_moving

        if download:
            if not os.path.exists(sdd_data_path):
                os.makedirs(sdd_data_path)
            if not os.path.exists(f"{sdd_data_path}/all_images.pkl"):
                print("Downloading SDD data")
                download_sdd_data(sdd_data_path)

        with open(f"{sdd_data_path}/all_images.pkl", "rb") as f:
            self.all_images = pickle.load(f)
        self.image = self.all_images[img_id]

        self.rescale_coordinates = rescale_coordinates

        if type(rescale_coordinates) is int:
            self.scale = calc_rescale(
                self.image.shape[0], self.image.shape[1], rescale_coordinates
            )
        elif rescale_coordinates:
            self.scale = calc_rescale(self.image.shape[0], self.image.shape[1], 10)
        else:
            self.scale = 1

        self.dequantized = dequantized
        if self.dequantized:
            with open(f"{sdd_data_path}/trajectories_dequantized.pkl", "rb") as f:
                self.trajectories = pickle.load(f)
            self.trajectories: dict[str, np.ndarray] = self.trajectories[img_id]
        else:
            with open(f"{sdd_data_path}/trajectories.pkl", "rb") as f:
                self.trajectories = pickle.load(f)
            self.trajectories: dict[str, np.ndarray] = self.trajectories[img_id]

        with open(f"{sdd_data_path}/ineqs.pkl", "rb") as f:
            self.all_ineqs = pickle.load(f)
        self.ineqs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = self.all_ineqs[
            img_id
        ]

        with open(f"{sdd_data_path}/polygons.pkl", "rb") as f:
            self.all_polygons = pickle.load(f)
        self.polygons = self.all_polygons[img_id]

    def get_scale(self):
        return self.scale

    def get_image(self):
        return self.image

    def get_trajectories(self):
        trajectories = self.trajectories
        if self.filter_moving:
            trajectories = dict(
                list(
                    filter_moving_trajectories(
                        dict(trajectories),
                    ).items()
                )
            )
        return trajectories

    def get_dataset(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """ "
        Returns the dataset as a tuple of three lists: train, validation and
        test. It returns the dataset as a list of tuples, where each tuple
        contains the trajectories in it's "raw" form, so as a list of points.
        """
        trajectories = list(self.get_trajectories().items())
        # Define the split percentages
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15

        # First split: training and remaining (validation + test)
        train_trajectories, remaining_trajectories = train_test_split(
            trajectories, train_size=train_size, random_state=42
        )
        train_trajectories: list[tuple[int, np.ndarray]]

        # Second split: validation and test
        val_trajectories, test_trajectories = train_test_split(
            remaining_trajectories,
            test_size=test_size / (val_size + test_size),
            random_state=42,
        )
        val_trajectories: list[tuple[int, np.ndarray]]
        test_trajectories: list[tuple[int, np.ndarray]]

        # Rescale the trajectories
        train_data = []
        for t_id, traj in train_trajectories:
            train_data.append(traj * self.scale)

        val_data = []
        for t_id, traj in val_trajectories:
            val_data.append(traj * self.scale)

        test_data = []
        for t_id, traj in test_trajectories:
            test_data.append(traj * self.scale)

        return train_data, val_data, test_data

    def get_trajectory_prediction_dataset(
        self,
        window_size: int = 5,
        sampling_rate: int = 70,
        predict_horizon_samples: int = 10,
    ) -> tuple[SampledHorizonDataset, Dataset, Dataset]:
        """
        Generates trajectory prediction datasets for training, validation, and testing.

        This method processes trajectory data to create datasets suitable for trajectory
        prediction tasks. It supports a specific configuration for precomputed datasets
        and dynamically generates datasets for other configurations.

        Args:
            window_size (int, optional): The size of the observation window in terms of
                the number of samples. Defaults to 5.
            sampling_rate (int, optional): The sampling rate for trajectory data.
                Defaults to 70.
            predict_horizon_samples (int, optional): The number of prediction horizon
                samples. Defaults to 10.

        Returns:
            tuple[SampledHorizonDataset, Dataset, Dataset]: A tuple containing the training,
                validation, and testing datasets. The training dataset is a
                `SampledHorizonDataset`, while the validation and testing datasets are
                `TensorDataset` objects.

        Raises:
            AssertionError: If the horizon distribution is set to "exponential", which is
                not implemented.

        Notes:
            - If the configuration matches the predefined "paper configuration", the method
              loads precomputed datasets from files.
            - For other configurations, the method dynamically splits the trajectory data
              into training, validation, and testing sets, and processes them into the
              required format.
            - Metadata for validation and testing datasets is stored in `self.metadata_val`
              and `self.metadata_test`, respectively.
        """
        is_paper_config = window_size == 5 and sampling_rate == 70
        is_paper_config = is_paper_config and self.img_id in [2, 12]
        is_paper_config = is_paper_config and self.dequantized and self.filter_moving
        # rescale_coordinates is bool and true
        is_paper_config = is_paper_config and (
            isinstance(self.rescale_coordinates, bool) and self.rescale_coordinates
        )
        is_paper_config = is_paper_config and predict_horizon_samples == 10
        if is_paper_config:
            paths = {
                2: f"{self.sdd_data_path}/static_dataset/sdd_dataset_2.pkl",
                12: f"{self.sdd_data_path}/static_dataset/sdd_dataset_12.pkl",
            }
            path = paths[self.img_id]
            sampled_horizon_kwargs = {
                "distribution": "mixture_uniform",
                "bin_size_mixture": sampling_rate,
            }
            train, val, test = load_sdd_trajectories_from_file(
                path, sampled_horizon_kwargs
            )
            return train, val, test

        trajectories = list(self.get_trajectories().items())

        min_length = (1 + window_size) * sampling_rate
        trajectories: list[tuple[str, np.ndarray]] = [
            (t_id, traj) for t_id, traj in trajectories if traj.shape[0] >= min_length
        ]

        # Define the split percentages
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15

        # First split: training and remaining (validation + test)
        train_trajectories, remaining_trajectories = train_test_split(
            trajectories, train_size=train_size, random_state=42
        )

        # Second split: validation and test
        val_trajectories, test_trajectories = train_test_split(
            remaining_trajectories,
            test_size=test_size / (val_size + test_size),
            random_state=42,
        )

        horizon_distribution = "mixture_uniform"

        # Convert trajectories to datasets
        X_train, y_train, metadata_train = trajectories_to_dataset_horizon(
            train_trajectories, window_size, sampling_rate
        )
        self.metadata_train = metadata_train

        X_val, y_val, metadata_val = trajectories_to_dataset_horizon(
            val_trajectories, window_size, sampling_rate
        )

        X_test, y_test, metadata_test = trajectories_to_dataset_horizon(
            test_trajectories, window_size, sampling_rate
        )

        X_train = torch.tensor(X_train).to(torch.float32)
        y_train = [torch.tensor(y).to(torch.float32) * self.scale for y in y_train]
        X_val = torch.tensor(X_val).to(torch.float32) * self.scale
        y_val = [torch.tensor(y).to(torch.float32) * self.scale for y in y_val]
        X_test = torch.tensor(X_test).to(torch.float32) * self.scale
        y_test = [torch.tensor(y).to(torch.float32) * self.scale for y in y_test]

        X_train = X_train.reshape(-1, window_size * 2)
        X_val = X_val.reshape(-1, window_size * 2)
        X_test = X_test.reshape(-1, window_size * 2)

        X_train_np = X_train.numpy()
        y_train_np = [y.numpy() for y in y_train]

        X_val_np = X_val.numpy()
        y_val_np = [y.numpy() for y in y_val]

        X_test_np = X_test.numpy()
        y_test_np = [y.numpy() for y in y_test]

        assert horizon_distribution != "exponential", "Not implemented yet"

        train_dataset = SampledHorizonDataset(
            X_train_np,
            y_train_np,
            distribution=horizon_distribution,
            bin_size_mixture=sampling_rate,
        )

        val_dataset_sampler = SampledHorizonDataset(
            X_val_np,
            y_val_np,
            distribution=horizon_distribution,
            bin_size_mixture=sampling_rate,
            return_index=True,
        )

        # set_seed
        np.random.seed(42)

        final_metadata_val = []
        final_val_X = []
        final_val_y = []
        for i in range(len(val_dataset_sampler)):
            for _ in range(predict_horizon_samples):
                x, y, idx = val_dataset_sampler[i]
                final_val_X.append(x)
                final_val_y.append(y)
                (t_id, idx_window, _) = metadata_val[i]
                final_metadata_val.append((t_id, idx_window, idx))
        self.metadata_val = final_metadata_val

        final_val_X_np = np.stack(final_val_X, axis=0)
        final_val_y_np = np.stack(final_val_y, axis=0)

        val_dataset = TensorDataset(
            torch.tensor(final_val_X_np), torch.tensor(final_val_y_np)
        )

        test_dataset_sampler = SampledHorizonDataset(
            X_test_np,
            y_test_np,
            distribution=horizon_distribution,
            bin_size_mixture=sampling_rate,
            return_index=True,
        )

        final_metadata_test = []
        final_test_X = []
        final_test_y = []
        for i in range(len(test_dataset_sampler)):
            for _ in range(predict_horizon_samples):
                x, y, idx = test_dataset_sampler[i]
                final_test_X.append(x)
                final_test_y.append(y)
                (t_id, idx_window, _) = metadata_test[i]
                final_metadata_test.append((t_id, idx_window, idx))
        self.metadata_test = final_metadata_test

        final_test_X_np = np.stack(final_test_X, axis=0)
        final_test_y_np = np.stack(final_test_y, axis=0)

        test_dataset = TensorDataset(
            torch.tensor(final_test_X_np), torch.tensor(final_test_y_np)
        )

        return train_dataset, val_dataset, test_dataset

    def get_ineqs(self, do_rescale=True) -> DNF:
        raw_ineqs = self.ineqs

        polytopes = []
        for constraint_class in self.constraint_classes:
            for A, b in raw_ineqs[constraint_class]:
                p = PolytopeH(A, b)
                if do_rescale:
                    p.rescale(self.scale)
                polytopes.append(p)

        return DNF(polytopes)

    def get_polygons(self, do_rescale=True) -> list[PolytopeV]:
        all_polygons = self.polygons
        polygons = []
        for constraint_class in self.constraint_classes:
            for vertices in all_polygons[constraint_class]:
                p = PolytopeV(np.array(vertices))
                if do_rescale:
                    p.rescale(self.scale)
                polygons.append(p)
        return polygons
