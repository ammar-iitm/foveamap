"""Golden-frame regression test verifying exact deterministic data ingestion and preprocessing."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from foveamap.core.contracts import LiDARFrame
from foveamap.core.config import PreprocessConfig
from foveamap.data.file import PcdFileSource
from foveamap.data.preprocess import LiDARPreprocessor


GOLDEN_PCD_CONTENT = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 4 2
TYPE F F F F I
COUNT 1 1 1 1 1
WIDTH 8
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 8
DATA ascii
0.0 0.0 1.73 0.05 0
0.2 0.3 1.73 0.10 1
5.0 2.0 -0.5 0.75 10
12.0 -3.5 0.2 0.85 24
25.0 8.0 1.5 0.90 32
45.0 -12.0 -1.0 0.60 48
120.0 0.0 0.0 0.95 50
15.0 5.0 12.0 0.40 60
"""

# Expected retained points after preprocessing:
# Point 0: range 0m -> filtered (min_range=1.0)
# Point 1: xy distance ~0.36m -> filtered (self-hit radius 1.0m)
# Point 6: range 120m -> filtered (max_range=100.0)
# Point 7: z=12.0m -> filtered (z_max=5.0)
# Points retained: 2, 3, 4, 5 (4 points)
EXPECTED_PTS = np.array([
    [5.0, 2.0, -0.5],
    [12.0, -3.5, 0.2],
    [25.0, 8.0, 1.5],
    [45.0, -12.0, -1.0],
], dtype=np.float32)

EXPECTED_INTENSITY = np.array([0.75, 0.85, 0.90, 0.60], dtype=np.float32)
EXPECTED_RING = np.array([10, 24, 32, 48], dtype=np.int16)


def test_golden_frame_ingestion_and_preprocessing(tmp_path: Path):
    pcd_path = tmp_path / "golden_sample.pcd"
    pcd_path.write_text(GOLDEN_PCD_CONTENT, encoding="ascii")

    # 1. Ingest via PcdFileSource
    source = PcdFileSource(pcd_path)
    assert len(source) == 1
    raw_frame = source[0]

    assert isinstance(raw_frame, LiDARFrame)
    assert raw_frame.num_points == 8
    assert raw_frame.frame_id == "golden_sample"

    # 2. Preprocess with explicit configuration
    pre_cfg = PreprocessConfig(
        min_range_m=1.0,
        max_range_m=100.0,
        z_min_m=-2.0,
        z_max_m=5.0,
        remove_self_hits=True,
        self_hit_radius_m=1.0,
    )
    preprocessor = LiDARPreprocessor(pre_cfg)
    processed_frame = preprocessor.process(raw_frame)

    # 3. Assert exact golden reference invariants
    assert processed_frame.num_points == 4
    np.testing.assert_allclose(processed_frame.pts, EXPECTED_PTS, atol=1e-6)
    np.testing.assert_allclose(processed_frame.intensity, EXPECTED_INTENSITY, atol=1e-6)
    np.testing.assert_array_equal(processed_frame.ring, EXPECTED_RING)
    assert processed_frame.pose.shape == (4, 4)
    np.testing.assert_array_equal(processed_frame.pose, np.eye(4))

    stats = processed_frame.metadata["preprocessing"]
    assert stats["original_points"] == 8
    assert stats["retained_points"] == 4
    assert stats["filtered_points"] == 4

    # 4. Assert determinism across replay
    processed_frame_2 = preprocessor.process(raw_frame)
    np.testing.assert_array_equal(processed_frame.pts, processed_frame_2.pts)
    np.testing.assert_array_equal(processed_frame.intensity, processed_frame_2.intensity)
    np.testing.assert_array_equal(processed_frame.ring, processed_frame_2.ring)
