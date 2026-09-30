import pytest

from foveamap.frames import sim_frames
from foveamap.sim import simulate_sequence, save_sequence


@pytest.fixture(scope="session")
def drive(tmp_path_factory):
    """A short simulated drive: (frames, truth)."""
    path = str(tmp_path_factory.mktemp("sim") / "drive.npz")
    save_sequence(path, simulate_sequence(seed=7, n_frames=4))
    return sim_frames(path)
