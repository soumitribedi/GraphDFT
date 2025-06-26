import os
import torch
import pytest
import math
from utils.geom_utils import read_geom_file, read_descriptor_file, compute_distance, compute_distance_coord

@pytest.fixture
def sample_geom(tmp_path):
    content = """\
H 0.0 0.0 0.0
C 1.0 0.0 0.0
O 0.0 1.0 0.0
"""
    p = tmp_path / "geom.xyz"
    p.write_text(content)
    return str(p)

def test_read_geom_file_basic(sample_geom):
    atoms = read_geom_file(sample_geom)
    # Expect three atoms with correct coordinates
    assert atoms == [
        ('H', (0.0, 0.0, 0.0)),
        ('C', (1.0, 0.0, 0.0)),
        ('O', (0.0, 1.0, 0.0)),
    ]

def test_read_geom_file_invalid_line(tmp_path):
    bad = tmp_path / "bad.xyz"
    bad.write_text("Xx 0.0 0.0\n")  # only 3 fields
    with pytest.raises(ValueError) as err:
        read_geom_file(str(bad))
    assert "expected 4 fields" in str(err.value)


def test_read_geom_file_bad_number(tmp_path):
    bad = tmp_path / "bad2.xyz"
    bad.write_text("H not_a_number 0.0 0.0\n")
    with pytest.raises(ValueError) as err:
        read_geom_file(str(bad))
    assert "could not parse coordinates" in str(err.value)


def test_read_geom_file_missing_file():
    with pytest.raises(IOError):
        read_geom_file("no_such_file.xyz")


@pytest.fixture
def sample_descriptor(tmp_path):
    content = (
        "Element,Other,A,B,C\n"
        "H,foo,1.0,2.0,3.0\n"
        "C,bar,4.5,5.5,6.5\n"
        "H,baz,7.0,8.0,9.0\n"
    )
    p = tmp_path / "desc.csv"
    p.write_text(content)
    return str(p)

def test_read_descriptor_file_basic(sample_descriptor):
    desc = read_descriptor_file(sample_descriptor)
    assert set(desc.keys()) == {"H1", "C1", "H2"}
    assert torch.allclose(desc["H1"], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(desc["C1"], torch.tensor([4.5, 5.5, 6.5], dtype=torch.float64))
    assert torch.allclose(desc["H2"], torch.tensor([7.0, 8.0, 9.0], dtype=torch.float64))

def test_read_descriptor_file_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")  # totally empty file
    result = read_descriptor_file(str(empty))
    assert result == {}

def test_read_descriptor_file_invalid_row(tmp_path):
    bad = tmp_path / "bad.csv"
    # header + one too-short row
    bad.write_text("Element,Other,A\nH,foo\n")
    with pytest.raises(ValueError) as exc:
        read_descriptor_file(str(bad))
    assert "expected at least 3 columns" in str(exc.value)

def test_read_descriptor_file_bad_value(tmp_path):
    bad = tmp_path / "bad2.csv"
    bad.write_text("Element,Other,A,B\nH,foo,not_a_number,2.0\n")
    with pytest.raises(ValueError) as exc:
        read_descriptor_file(str(bad))
    assert "could not parse descriptor values" in str(exc.value)

def test_read_descriptor_file_missing_file():
    with pytest.raises(FileNotFoundError):
        read_descriptor_file("nonexistent.csv")

def test_compute_distance_zero():
    """Distance between identical points is exactly zero."""
    atom = ("X", (1.23, 4.56, 7.89))
    dist = compute_distance(atom, atom)
    assert isinstance(dist, torch.Tensor)
    assert dist.ndim == 0
    assert dist.dtype == torch.float64
    assert torch.isclose(dist, torch.tensor(0.0, dtype=torch.float64))

@pytest.mark.parametrize("p1,p2,expected", [
    (("H", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0)), 1.0),
    (("A", (0.0, 0.0, 0.0)), ("B", (0.0, 3.0, 4.0)), 5.0),
    (("C", (1.0, 2.0, 2.0)), ("D", (4.0, 6.0, 6.0)), math.sqrt(41)),
])
def test_compute_distance_known(p1, p2, expected):
    """Check a few known-distance cases in 3D."""
    dist = compute_distance(p1, p2)
    assert torch.isclose(dist, torch.tensor(expected, dtype=torch.float64), atol=1e-7)

def test_known_distances_2d_and_3d():
    # 2D: (0,0) -> (3,4) = 5
    c1 = torch.tensor([0.0, 0.0], dtype=torch.float64)
    c2 = torch.tensor([3.0, 4.0], dtype=torch.float64)
    d = compute_distance_coord(c1, c2)
    assert d.ndim == 0
    assert d.dtype == torch.float64
    assert torch.isclose(d, torch.tensor(5.0, dtype=torch.float64))

    # 3D: (1,2,2) -> (4,6,6) = sqrt(41)
    c1 = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)
    c2 = torch.tensor([4.0, 6.0, 6.0], dtype=torch.float32)
    d = compute_distance_coord(c1, c2)
    assert d.dtype == torch.float32
    assert math.isclose(d.item(), math.sqrt(41), rel_tol=1e-6)


def test_shape_mismatch_raises_runtime_error():
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 2.0])  # incompatible
    with pytest.raises(RuntimeError):
        compute_distance_coord(a, b)


def test_non_tensor_inputs_raise_type_error():
    # Passing a list or tuple instead of tensor
    with pytest.raises(TypeError):
        compute_distance_coord([0.0, 0.0, 0.0], torch.tensor([0.0, 0.0, 0.0]))
    with pytest.raises(TypeError):
        compute_distance_coord(torch.tensor([0.0, 0.0, 0.0]), (1.0, 2.0, 3.0))