import sdd.constrained_sdd as csdd
import tempfile
import os


def test_download_sdd_data():
    folder = tempfile.mkdtemp()
    csdd.download_sdd_data(folder)
    assert len(os.listdir(folder)) > 0