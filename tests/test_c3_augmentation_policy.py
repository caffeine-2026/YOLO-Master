"""Regression tests for the C3 industrial-image augmentation protocol."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from smoke.c3.augmentation.tools.freeze_selection import summary
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import IndustrialPhotometric, RandomFlip, RandomPerspective
from ultralytics.utils.instance import Instances

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "smoke/c3/augmentation/config/augmentation_protocol.yaml"


def test_industrial_photometric_defaults_are_disabled_and_validated():
    cfg = get_cfg()
    assert cfg.industrial_contrast == 0.0
    assert cfg.industrial_blur_prob == 0.0
    assert cfg.industrial_blur_sigma == 0.0
    assert cfg.industrial_noise_prob == 0.0
    assert cfg.industrial_noise_std == 0.0
    with pytest.raises(ValueError, match="industrial_contrast"):
        get_cfg(overrides={"industrial_contrast": 1.1})
    with pytest.raises(TypeError, match="industrial_noise_std"):
        get_cfg(overrides={"industrial_noise_std": "4.0"})


def test_photometric_augmentation_preserves_grayscale_and_annotations():
    base = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    image = np.repeat(base[..., None], 3, axis=2)
    boxes = Instances(np.array([[0.4, 0.6, 0.2, 0.3]], dtype=np.float32), bbox_format="xywh", normalized=True)
    classes = np.array([[2.0]], dtype=np.float32)
    labels = {"img": image, "instances": boxes, "cls": classes}
    before_boxes = boxes.bboxes.copy()
    np.random.seed(824)
    transform = IndustrialPhotometric(contrast=0.2, blur_prob=1.0, blur_sigma=0.8, noise_prob=1.0, noise_std=4.0)
    result = transform.apply_image(
        labels,
        {
            "contrast_factor": 1.1,
            "apply_blur": True,
            "blur_sigma": 0.8,
            "apply_noise": True,
            "noise_std": 4.0,
        },
    )
    assert result["img"].dtype == np.uint8
    assert np.array_equal(result["img"][..., 0], result["img"][..., 1])
    assert np.array_equal(result["img"][..., 0], result["img"][..., 2])
    assert result["instances"] is boxes
    assert np.array_equal(result["instances"].bboxes, before_boxes)
    assert np.array_equal(result["cls"], classes)


def test_horizontal_flip_updates_normalized_box_exactly():
    instances = Instances(
        np.array([[0.25, 0.6, 0.2, 0.3]], dtype=np.float32),
        segments=np.zeros((1, 0, 2), dtype=np.float32),
        bbox_format="xywh",
        normalized=True,
    )
    labels = {"img": np.zeros((20, 40, 3), dtype=np.uint8), "instances": instances, "cls": np.array([[0]])}
    result = RandomFlip(p=1.0, direction="horizontal")(labels)
    assert np.allclose(result["instances"].bboxes, [[0.75, 0.6, 0.2, 0.3]])


def test_random_perspective_identity_preserves_boxes():
    transform = RandomPerspective(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0)
    boxes = np.array([[10.0, 20.0, 40.0, 60.0], [100.0, 120.0, 150.0, 180.0]], dtype=np.float32)
    assert np.array_equal(transform.apply_bboxes(boxes, np.eye(3, dtype=np.float32)), boxes)


def test_random_perspective_translation_updates_box_corners_exactly():
    transform = RandomPerspective(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0)
    boxes = np.array([[10.0, 20.0, 40.0, 60.0]], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert np.array_equal(transform.apply_bboxes(boxes, matrix), [[15.0, 23.0, 45.0, 63.0]])


def test_protocol_prevents_test_leakage_and_destructive_augmentations():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["base_commit"] == "bf6c7c508635dec0be849aedaa3eac5d88ed220d"
    assert protocol["leakage_guard"]["locked_test_access_before_freeze"] == "prohibited"
    assert protocol["search"]["initial_seed"] == 824
    assert protocol["locked_training"]["seeds"] == [824, 825, 826]
    disabled = protocol["common_disabled_augmentations"]
    for key in ("hsv_h", "hsv_s", "mosaic", "mixup", "cutmix", "copy_paste"):
        assert disabled[key] == 0.0
    for dataset in ("neu", "deeppcb"):
        for sample_size in (10, 50, 100, 500):
            data = yaml.safe_load(
                (ROOT / f"smoke/c3/augmentation/config/data/{dataset}_{sample_size}_validation_only.yaml").read_text()
            )
            assert "test" not in data
        assert set(protocol["policies"][dataset]) == {"baseline", "mild", "medium", "strong"}


def test_deeppcb_geometry_is_bounded_below_neu_for_each_strength():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    for policy in ("mild", "medium", "strong"):
        neu = protocol["policies"]["neu"][policy]
        deeppcb = protocol["policies"]["deeppcb"][policy]
        assert deeppcb["degrees"] < neu["degrees"]
        assert deeppcb["translate"] < neu["translate"]
        assert deeppcb["scale"] < neu["scale"]


def test_three_seed_interval_uses_paired_t_distribution():
    result = summary([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["sample_std"] == 1.0
    assert result["ci95_lower"] == pytest.approx(-0.484138, abs=1e-6)
    assert result["ci95_upper"] == pytest.approx(4.484138, abs=1e-6)
