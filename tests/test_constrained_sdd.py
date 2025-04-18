import sdd.constrained_sdd as csdd
import tempfile
import os
import numpy as np


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
    train, val, test, metadata = csdd.load_sdd_trajectories_from_file(
        static_12_path, sampled_horizon_kwargs
    )


def test_full_horizon():
    folder = "data/sdd"
    sdd = csdd.ConstrainedStanfordDroneDataset(12, sdd_data_path=folder, download=True)
    assert sdd.polygons is not None
    sdd.get_dataset()

    train, val, test = sdd.get_trajectory_prediction_dataset()

    train_full_horizon, val_full_horizon, test_full_horizon = sdd.get_trajectories_prediction_full_horizon()

    for _ in range(10):
        idx_train = np.random.randint(0, len(train_full_horizon))
        x, _ = train_full_horizon[idx_train]
        x = x.reshape(-1)
        x_ref, _ = train[idx_train]
        assert x.shape == x_ref.shape
        assert np.allclose(x, x_ref, atol=1e-5)

        idx_val = np.random.randint(0, len(val_full_horizon))
        x, _ = val_full_horizon[idx_val]
        x = x.reshape(-1)
        x_ref, _ = val[idx_val]
        assert x.shape == x_ref.shape
        assert np.allclose(x, x_ref, atol=1e-5)

        idx_test = np.random.randint(0, len(test_full_horizon))
        x, _ = test_full_horizon[idx_test]
        x = x.reshape(-1)
        x_ref, _ = test[idx_test]
        assert x.shape == x_ref.shape
        assert np.allclose(x, x_ref, atol=1e-5)
