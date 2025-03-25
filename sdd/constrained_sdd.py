import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os

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


class DNF:
    """
    Represents a Disjunctive Normal Form (DNF) consisting of a list of polytopes.

    Attributes:
        polytopes (list[PolytopeH]): A list of PolytopeH objects that define the DNF.
    """

    def __init__(self, polytopes: list[PolytopeH]):
        self.polytopes = polytopes


class PolygonV:
    """
    A class representing a polygon defined by its vertices.

    Attributes:
        vertices (np.ndarray): A NumPy array containing the vertices of the polygon.
                                Each vertex is typically represented as a coordinate pair (x, y).
    """

    def __init__(self, vertices: np.ndarray):
        self.vertices = vertices


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
        trajectories = self.get_trajectories()
        # Define the split percentages
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15

        # First split: training and remaining (validation + test)
        train_trajectories, remaining_trajectories = train_test_split(
            trajectories, train_size=train_size, random_state=42
        )
        train_trajectories = train_trajectories

        # Second split: validation and test
        val_trajectories, test_trajectories = train_test_split(
            remaining_trajectories,
            test_size=test_size / (val_size + test_size),
            random_state=42,
        )
        val_trajectories = val_trajectories
        test_trajectories = test_trajectories

        return train_trajectories, val_trajectories, test_trajectories

    def get_ineqs(self) -> DNF:
        raw_ineqs = self.ineqs

        polytopes = []
        for constraint_class in self.constraint_classes:
            for A, b in raw_ineqs[constraint_class]:
                polytopes.append(PolytopeH(A, b))

        return DNF(polytopes)

    def get_polygons(self) -> list[PolygonV]:
        all_polygons = self.polygons
        polygons = []
        for constraint_class in self.constraint_classes:
            for vertices in all_polygons[constraint_class]:
                polygons.append(PolygonV(np.array(vertices)))
        return polygons
