"""Training-recipe helpers (no training is run)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train import augment, balanced_frame_probs, top_confusions  # noqa: E402
from foveamap.model import RangeUNet  # noqa: E402


def test_balanced_sampling_favours_rare_class_frames():
    Y = np.zeros((3, 4, 4), np.int8)            # class 0 everywhere
    Y[1, 0, :2] = 2                             # frame 1 has two pixels of a rare class
    Y[2] = -1                                   # frame 2 is unlabelled apart from one pixel
    Y[2, 0, 0] = 0
    p = balanced_frame_probs(Y, np.array([1.0, 1.0, 10.0]))
    np.testing.assert_allclose(p.sum(), 1.0)
    assert p[1] > p[0] > p[2] > 0


def test_augment_scales_geometry_and_keeps_empty_pixels_empty():
    torch.manual_seed(0)
    xb = torch.rand(4, 8, 2, 6)
    xb[:, 5] = (xb[:, 5] > 0.3).float()
    rb = xb[:, 0] * 50
    xa, ra = augment(xb, rb)
    ratio = xa[:, 0:4] / xb[:, 0:4]
    for b in range(4):
        s = ratio[b].flatten()
        assert torch.allclose(s, s[0].expand_as(s), atol=1e-5) and 0.95 <= s[0] <= 1.05
        assert torch.allclose(ra[b] / rb[b], s[0].expand_as(ra[b]), atol=1e-5)
    assert torch.all(xa[:, 4][xb[:, 5] == 0] == 0) and xa[:, 4].max() <= 1
    assert torch.equal(xa[:, 5:], xb[:, 5:])     # valid mask and motion residuals untouched
    assert not torch.equal(xa, xb) and xb.data_ptr() != xa.data_ptr()


def test_reset_head_changes_only_the_class_layer():
    torch.manual_seed(0)
    m = RangeUNet()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.sem.reset_parameters()
    changed = {k for k, v in m.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {"sem.weight", "sem.bias"}


def test_top_confusions():
    tc = top_confusions([[90, 10, 0], [30, 60, 10], [0, 0, 0]], ["a", "b", "c"], [True, True, False])
    assert tc["a"] == dict(recall=0.9, predicted_as=[("b", 0.1)])
    assert tc["b"]["recall"] == 0.6 and tc["b"]["predicted_as"][0] == ("a", 0.3)
    assert "c" not in tc


def test_training_arrays_on_disk_match_ram(drive, tmp_path):
    from foveamap.frames import SIM_INFO, frames_to_training_arrays
    frames, _ = drive
    ram = frames_to_training_arrays(frames, SIM_INFO)
    disk = frames_to_training_arrays(frames, SIM_INFO, out_dir=str(tmp_path))
    for a, b in zip(ram, disk):
        assert isinstance(b, np.memmap)
        np.testing.assert_array_equal(a, b)
    from train import label_stats
    from foveamap.sim import NUM_CLASSES
    freq, n_mov, n_static = label_stats(disk[1], disk[2], NUM_CLASSES, chunk=1)
    Y, M = ram[1], ram[2]
    assert (freq == np.bincount(Y[Y >= 0].astype(np.int64), minlength=NUM_CLASSES)).all()
    assert n_mov == M.sum() and n_static == (~M & (Y >= 0)).sum()


def test_wider_network_saves_and_loads_at_its_width(tmp_path):
    from foveamap.model import RangeUNet, load_model, widths_for, widths_of
    m = RangeUNet(widths_for(2)).eval()
    assert widths_for(2) == (32, 64, 128, 192) and widths_for(1) == (16, 32, 64, 96)
    path = tmp_path / "wide.pt"
    torch.save(m.state_dict(), path)
    back = load_model(str(path), torch.device("cpu"))
    assert widths_of(back.state_dict()) == widths_for(2)
    x = torch.randn(1, 8, 16, 64)
    with torch.no_grad():
        sem, mot = m(x)
        sem2, mot2 = back(x)
    assert sem.shape == (1, 9, 16, 64) and mot.shape == (1, 16, 64)
    assert torch.equal(sem, sem2) and torch.equal(mot, mot2)


def test_standard_checkpoint_keeps_its_shape():
    from foveamap.model import load_model, widths_of, BASE_WIDTHS
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "range_unet.pt")
    assert widths_of(load_model(ckpt, torch.device("cpu")).state_dict()) == BASE_WIDTHS
