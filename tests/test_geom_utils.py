import os
import torch
import pytest
from utils.geom_utils import read_geom_file

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

