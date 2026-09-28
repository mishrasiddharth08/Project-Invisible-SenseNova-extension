import numpy as np
from PIL import Image

from pi_sensenova import degrid


def load_tests(loader, tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith("test_") and callable(fn))


def _checker(width=256, height=256, amplitude=4):
    yy, xx = np.indices((height, width))
    pattern = np.where((xx + yy) % 2, amplitude, -amplitude)
    array = np.clip(128 + pattern, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(array[:, :, None], 3, axis=2), "RGB")


def _phase_amplitude(image):
    array = np.asarray(image, dtype=np.float32)[..., 0]
    phases = [array[y::2, x::2].mean() for y in (0, 1) for x in (0, 1)]
    return max(phases) - min(phases)


def test_disabled_returns_original_object():
    image = _checker()
    assert degrid.apply(image) is image


def test_clean_image_is_exact_passthrough():
    image = Image.new("RGB", (257, 259), (91, 121, 151))
    image.info["parameters"] = "kept"
    before = image.tobytes()
    result, stats = degrid.apply_with_stats(image, enabled=True)
    assert result is image
    assert result.tobytes() == before
    assert result.info["parameters"] == "kept"
    assert stats.skipped


def test_checker_grid_is_reduced_without_mutating_input():
    image = _checker()
    before = image.tobytes()
    result, stats = degrid.apply_with_stats(image, enabled=True)
    assert image.tobytes() == before
    assert not stats.skipped
    assert _phase_amplitude(result) < _phase_amplitude(image) * 0.15


def test_hard_edge_is_preserved():
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    array[:, 128:] = 255
    image = Image.fromarray(array, "RGB")
    result, _ = degrid.apply_with_stats(image, enabled=True)
    output = np.asarray(result)
    assert output[:, 100].max() == 0
    assert output[:, 155].min() == 255


def test_large_tiled_result_has_no_tile_seams():
    image = _checker(1031, 1027)
    tiled, stats = degrid._apply_tiled(image)
    full, _ = degrid._apply_full(image)
    assert stats.tiled
    difference = np.abs(np.asarray(tiled, dtype=np.int16) - np.asarray(full, dtype=np.int16))
    assert difference.max() <= 1


def test_tiny_image_is_unchanged():
    image = _checker(8, 8)
    assert degrid.apply(image, enabled=True) is image
