import os
import glob
import pytest


@pytest.fixture(autouse=True)
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)
    yield
    for f in glob.glob("data/test_*"):
        try:
            os.remove(f)
        except OSError:
            pass
