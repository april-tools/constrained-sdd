import sdd.constrained_sdd as csdd
import tempfile
import os


def test_download_sdd_data():
    folder = tempfile.mkdtemp()
    csdd.download_sdd_data(folder)
    assert len(os.listdir(folder)) > 0


def test_download_and_load_sdd_data():
    folder = tempfile.mkdtemp()
    sdd = csdd.ConstrainedStanfordDroneDataset(0, sdd_data_path=folder, download=True)
    assert sdd.polygons is not None
