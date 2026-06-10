"""Data adapters — toy dataset, Cityscapes remap, and joint transforms."""
import pytest
import torch

from fiedler.data import (
    ToyDrivingDataset, make_toy_scene, CityscapesDataset,
    CITYSCAPES_ID_TO_TRAINID, build_trainid_lut, remap_to_trainids,
)
from fiedler.data import transforms as T


# --------------------------------------------------------------------------- #
# toy dataset
# --------------------------------------------------------------------------- #
def test_toy_scene_shapes_and_ranges():
    img, label = make_toy_scene(96, 192, seed=0)
    assert img.shape == (3, 96, 192) and label.shape == (96, 192)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert label.min() >= 0 and label.max() <= 18           # valid trainIds


def test_toy_scene_is_deterministic_per_seed():
    a_img, a_lab = make_toy_scene(64, 64, seed=7)
    b_img, b_lab = make_toy_scene(64, 64, seed=7)
    c_img, _ = make_toy_scene(64, 64, seed=8)
    assert torch.equal(a_lab, b_lab) and torch.allclose(a_img, b_img)
    assert not torch.allclose(a_img, c_img)                 # different seed -> different


def test_toy_dataset_len_and_items():
    ds = ToyDrivingDataset(n=5, size=(64, 128), seed=0)
    assert len(ds) == 5
    img, label = ds[2]
    assert img.shape == (3, 64, 128) and label.shape == (64, 128)


def test_toy_dataset_has_multiple_classes():
    _, label = ToyDrivingDataset(n=1, size=(128, 256))[0]
    assert label.unique().numel() >= 3                       # sky/building/road at least


# --------------------------------------------------------------------------- #
# Cityscapes remap
# --------------------------------------------------------------------------- #
def test_trainid_lut_maps_known_ids():
    lut = build_trainid_lut()
    assert lut[7].item() == 0       # road
    assert lut[26].item() == 13     # car
    assert lut[33].item() == 18     # bicycle


def test_trainid_lut_ignores_unmapped():
    lut = build_trainid_lut(ignore_index=255)
    for raw in (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30):
        assert lut[raw].item() == 255


def test_remap_tensor():
    raw = torch.tensor([[7, 26], [33, 0]])      # road, car, bicycle, unlabeled
    out = remap_to_trainids(raw)
    assert out.tolist() == [[0, 13], [18, 255]]
    assert out.dtype == torch.long


def test_remap_covers_all_19_classes():
    assert len(set(CITYSCAPES_ID_TO_TRAINID.values())) == 19
    assert sorted(CITYSCAPES_ID_TO_TRAINID.values()) == list(range(19))


def test_cityscapes_missing_root_raises_helpful_error():
    with pytest.raises(FileNotFoundError, match="datasets/README"):
        CityscapesDataset(root="datasets/does_not_exist", split="train")


# --------------------------------------------------------------------------- #
# joint transforms
# --------------------------------------------------------------------------- #
def _pair(H=64, W=96):
    return torch.rand(3, H, W), torch.randint(0, 19, (H, W))


def test_random_crop_size_and_alignment():
    img, lab = _pair(64, 96)
    cimg, clab = T.RandomCrop((32, 48))(img, lab)
    assert cimg.shape == (3, 32, 48) and clab.shape == (32, 48)


def test_random_crop_pads_when_too_small():
    img, lab = _pair(20, 20)
    cimg, clab = T.RandomCrop((32, 32), ignore_index=255)(img, lab)
    assert cimg.shape == (3, 32, 32) and clab.shape == (32, 32)
    assert (clab == 255).any()                               # padding marked ignore


def test_hflip_flips_both_consistently():
    img = torch.arange(2 * 3 * 4).float().reshape(2, 3, 4)   # distinct values
    lab = torch.arange(3 * 4).reshape(3, 4)
    fimg, flab = T.RandomHorizontalFlip(p=1.0)(img, lab)
    assert torch.equal(fimg, torch.flip(img, dims=[-1]))
    assert torch.equal(flab, torch.flip(lab, dims=[-1]))


def test_resize_keeps_label_integer():
    img, lab = _pair(40, 40)
    rimg, rlab = T.Resize((20, 30))(img, lab)
    assert rimg.shape == (3, 20, 30) and rlab.shape == (20, 30)
    assert rlab.dtype == torch.long
    assert set(rlab.unique().tolist()).issubset(set(lab.unique().tolist()))  # no new labels


def test_normalize_changes_image_not_label():
    img, lab = _pair()
    nimg, nlab = T.Normalize()(img, lab)
    assert not torch.allclose(nimg, img)
    assert torch.equal(nlab, lab)


def test_compose_chains():
    img, lab = _pair(64, 96)
    tf = T.Compose([T.RandomHorizontalFlip(1.0), T.RandomCrop((32, 32)), T.Normalize()])
    oimg, olab = tf(img, lab)
    assert oimg.shape == (3, 32, 32) and olab.shape == (32, 32)
