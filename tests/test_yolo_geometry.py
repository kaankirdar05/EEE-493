import numpy as np

from vuhf.detectors.yolo import box_to_pixels, image_width, pixels_to_box, spectrogram_image, yolo_label_lines
from vuhf.geometry import Box, SpecGeometry

GEOM = SpecGeometry(262144, 512, 256)


def test_image_shape_is_square_for_default_geometry():
    P = np.random.default_rng(0).exponential(size=GEOM.shape)
    img = spectrogram_image(P)
    assert img.shape == (512, image_width(GEOM)) == (512, 512)
    assert img.dtype == np.uint8


def test_box_pixel_roundtrip():
    b = Box(t0=40000.0, t1=120000.0, f0=-0.2, f1=0.05)
    back = pixels_to_box(*box_to_pixels(b, GEOM), GEOM)
    assert np.allclose([back.t0, back.t1, back.f0, back.f1], [b.t0, b.t1, b.f0, b.f1])


def test_pixels_match_cell_edges():
    # cells (k0..k1, j0..j1) -> box -> pixels must give the cell edges (two frames per column)
    box = GEOM.cells_to_box(100, 199, 200, 399)
    x0, y0, x1, y1 = box_to_pixels(box, GEOM)
    assert np.allclose([x0, x1, y0, y1], [100.0, 200.0, 100.0, 200.0])


def test_signal_is_bright_in_the_image():
    rng = np.random.default_rng(1)
    P = rng.exponential(size=GEOM.shape)
    P[100:200, 200:400] *= 30.0
    img = spectrogram_image(P)
    assert img[100:200, 100:200].mean() > img[300:400, 300:400].mean() + 80


def test_label_lines_are_normalised_and_skip_thin_boxes():
    boxes = [Box(0, 131072, -0.5, 0.0), Box(1000, 1100, 0.1, 0.1 + 1e-5)]
    lines = yolo_label_lines(boxes, GEOM)
    assert len(lines) == 1
    cls, xc, yc, w, h = lines[0].split()
    assert cls == "0" and 0 <= float(xc) <= 1 and 0 <= float(yc) <= 1 and 0 < float(w) <= 1 and 0 < float(h) <= 1
