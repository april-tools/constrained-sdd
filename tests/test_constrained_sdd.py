import sdd.constrained_sdd as csdd
import tempfile
import os


def test_download_sdd_data():
    folder = tempfile.mkdtemp()
    csdd.download_sdd_data(folder)
    assert len(os.listdir(folder)) > 0


def test_download_and_load_sdd_data():
    folder = "data/sdd"
    sdd = csdd.ConstrainedStanfordDroneDataset(0, sdd_data_path=folder, download=True)
    assert sdd.polygons is not None
    sdd.get_dataset()

    sdd.get_trajectory_prediction_dataset(10, 10)

    static_12_path = f"{sdd.sdd_data_path}/static_dataset/sdd_dataset_12.pkl"

    sampled_horizon_kwargs = {
        "distribution": "mixture_uniform",
        "bin_size_mixture": 70,
    }
    train, val, test = csdd.load_sdd_trajectories_from_file(
        static_12_path, sampled_horizon_kwargs
    )