import torch
import pytest
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from utils.grid_utils import load_vxc_coordinates, load_density, cartesian_to_spherical, real_spherical_harmonics_on_grid, \
    element_atomic_numbers, radial_function, nuclear_charge_moment

@pytest.fixture
def tmp_vxc_file(tmp_path):
    content = "\n".join([
        "0.0 1.0 2.0 0.5 extra",    # valid
        "3.0 4.0 5.0 1.2",          # valid
        "bad line here",            # too few columns → skip
        "6.0 seven 8.0 0.3",        # non-numeric y → skip
        "9.0 10.0 11.0 0.4 trailing text"  # valid, extra fields OK
    ])
    p = tmp_path / "coords.dat"
    p.write_text(content)
    return str(p)

def test_load_vxc_coordinates_basic(tmp_vxc_file):
    coords = load_vxc_coordinates(tmp_vxc_file)
    # should parse 3 valid lines
    assert isinstance(coords, torch.Tensor)
    assert coords.dtype == torch.float64
    assert coords.shape == (3, 3)
    # check first and last rows
    assert torch.allclose(coords[0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.allclose(coords[2], torch.tensor([9.0, 10.0, 11.0], dtype=torch.float64))

def test_load_vxc_coordinates_all_invalid(tmp_path):
    p = tmp_path / "empty.dat"
    p.write_text("\n".join([
        "too few cols",
        "a b c d",
        "1.0 2.0"   # still too few cols
    ]))
    coords = load_vxc_coordinates(str(p))
    assert isinstance(coords, torch.Tensor)
    # no valid lines → shape (0,3)
    assert coords.shape == (0, 3)

def test_load_vxc_coordinates_missing_file():
    with pytest.raises(OSError):
        load_vxc_coordinates("this_file_does_not_exist.xyz")

@pytest.fixture
def tmp_density_file(tmp_path):
    content = "\n".join([
        "0.0 1.0 2.0 0.5 extra",       # valid → 0.5
        "3.0 4.0 5.0 1.2",             # valid → 1.2
        "too few cols",                # skip
        "6.0 7.0 8.0 not_a_number",    # skip
        "9.0 10.0 11.0 2.3 trailing"   # valid → 2.3
    ])
    p = tmp_path / "density.dat"
    p.write_text(content)
    return str(p)

def test_load_density_basic(tmp_density_file):
    vals = load_density(tmp_density_file)
    # Expect three entries in order
    assert isinstance(vals, torch.Tensor)
    assert vals.dtype == torch.float64
    assert vals.shape == (3,)
    assert torch.allclose(vals, torch.tensor([0.5, 1.2, 2.3], dtype=torch.float64))

def test_load_density_all_invalid(tmp_path):
    p = tmp_path / "bad.dat"
    p.write_text("\n".join([
        "a b c",               # too few cols
        "1 2 3 not_a_number",  # non-numeric
        "just three fields"    # too few cols
    ]))
    vals = load_density(str(p))
    assert isinstance(vals, torch.Tensor)
    assert vals.shape == (0,)  # empty

def test_load_density_missing_file():
    with pytest.raises(OSError):
        load_density("no_such_file.xyz")

@pytest.mark.parametrize("coords, expected", [
    # (x,y,z) -> (r, θ, φ)
    ((1.0, 0.0, 0.0), (1.0, math.pi/2, 0.0)),
    ((0.0, 1.0, 0.0), (1.0, math.pi/2, math.pi/2)),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),  # φ = atan2(0,0) → 0 by convention
    ((1.0, 1.0, 1.0), (math.sqrt(3), math.acos(1/math.sqrt(3)), math.pi/4)),
])
def test_scalar_inputs(coords, expected):
    x, y, z = coords
    r, θ, φ = cartesian_to_spherical(x, y, z, device="cpu")
    er, et, ep = expected
    assert isinstance(r, torch.Tensor) and r.ndim == 0
    assert r.dtype == torch.float64
    assert torch.isclose(r, torch.tensor(er, dtype=torch.float64), atol=1e-7)
    assert torch.isclose(θ, torch.tensor(et, dtype=torch.float64), atol=1e-7)
    assert torch.isclose(φ, torch.tensor(ep, dtype=torch.float64), atol=1e-7)

def test_vector_inputs_list_and_numpy():
    # two points at once via list input
    x = [1.0, 0.0]
    y = [0.0, 1.0]
    z = [0.0, 0.0]
    r, θ, φ = cartesian_to_spherical(x, y, z, device="cpu")
    # Expect [1,1], θ both π/2, φ [0, π/2]
    assert r.shape == (2,)
    assert torch.allclose(r, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(θ, torch.tensor([math.pi/2, math.pi/2], dtype=torch.float64))
    assert torch.allclose(φ, torch.tensor([0.0, math.pi/2], dtype=torch.float64))

    # numpy array input
    x_np = np.array([0.0, 0.0])
    y_np = np.array([0.0, 0.0])
    z_np = np.array([1.0, -1.0])
    r2, θ2, φ2 = cartesian_to_spherical(x_np, y_np, z_np, device="cpu")
    # r = [1,1], θ = [0, π], φ = [0, 0]
    assert torch.allclose(r2, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(θ2, torch.tensor([0.0, math.pi], dtype=torch.float64))
    assert torch.allclose(φ2, torch.tensor([0.0, 0.0], dtype=torch.float64))

def test_device_placement():
    if torch.cuda.is_available():
        dev = "cuda"
        r, θ, φ = cartesian_to_spherical(1.0, 2.0, 2.0, device=dev)
        assert r.device.type == "cuda"
        assert θ.device.type == "cuda"
        assert φ.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")


def test_zero_vector_behavior():
    """
    At the origin, r = 0 leads to θ = NaN and φ = 0 (by convention).
    """
    r, θ, φ = cartesian_to_spherical(0.0, 0.0, 0.0, device="cpu")
    assert torch.isclose(r, torch.tensor(0.0, dtype=torch.float64))
    assert torch.isnan(θ)
    # atan2(0,0) is defined as 0
    assert torch.isclose(φ, torch.tensor(0.0, dtype=torch.float64))

def test_l0_m0_constant():
    # Y_0^0 = 1/(2*sqrt(pi)) at any point
    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 3.0],
        [-1.5, 0.5, 0.5]
    ])
    x, y, z = pts.T
    out = real_spherical_harmonics_on_grid(0, 0, x, y, z, device="cpu")
    assert out.shape == (3,)
    c = 1.0/(2*math.sqrt(math.pi))
    assert torch.allclose(out, torch.full((3,), c, dtype=torch.float64), atol=1e-7)

@pytest.mark.parametrize("l,m", [
    (1, 0),
    (1, 1),
    (1, -1),
])
def test_l1_known_values(l, m):
    # at x=1,y=1,z=0 → θ=π/2, φ=π/4
    x = np.array([1.0])
    y = np.array([1.0])
    z = np.array([0.0])
    theta = math.pi/2
    phi   = math.pi/4

    out = real_spherical_harmonics_on_grid(l, m, x, y, z, device="cpu")
    val = out.item()

    if (l, m) == (1, 0):
        expected = math.sqrt(3/(4*math.pi)) * math.cos(theta)
    elif (l, m) == (1, 1):
        # positive instead of negative
        expected =  math.sqrt(3/(4*math.pi)) * math.sin(theta) * math.sin(phi)
    else:  # (1, -1)
        expected =  math.sqrt(3/(4*math.pi)) * math.sin(theta) * math.cos(phi)

    assert pytest.approx(val, rel=1e-6) == expected

def test_shape_and_dtype_and_device():
    x = [0.0, 0.0]
    y = [0.0, 1.0]
    z = [1.0, 0.0]
    out = real_spherical_harmonics_on_grid(2, 1, x, y, z, device="cpu")
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float64
    assert out.device.type == "cpu"
    assert out.shape == (2,)

def test_cuda_placement_if_available():
    if torch.cuda.is_available():
        out = real_spherical_harmonics_on_grid(0, 0, 0.0, 0.0, 1.0, device="cuda")
        assert out.device.type == "cuda"
    else:
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def patch_atomic_numbers(monkeypatch):
    """
    Ensure that element_atomic_numbers has the entries we expect for H and He.
    """
    monkeypatch.setitem(element_atomic_numbers, 'H', 1)
    monkeypatch.setitem(element_atomic_numbers, 'He', 2)

def test_h_atom_n1_matches_exp():
    """
    For hydrogen (Z=1) and n=1, ζ=1 and R_n = exp(-r).
    """
    R = [0.0, 1.0, 2.0]
    out = radial_function('H', R, n=1, device='cpu')
    expected = torch.tensor([math.exp(-r) for r in R], dtype=torch.float64)

    # dtype and device
    assert out.dtype == torch.float64
    assert out.device.type == 'cpu'

    # values
    torch.testing.assert_allclose(out, expected, rtol=1e-7, atol=0)

def test_he_atom_n2_manual():
    """
    For helium (Z=2) and n=2, ζ = (2 - 0.35*(2-1))/2 = 0.825
    R_n = r^(2-1) * exp(-0.825*r)
    """
    R = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    zeta = 0.825
    expected = R.pow(1) * torch.exp(-zeta * R)

    out = radial_function('He', R, n=2, device='cpu')
    assert out.shape == R.shape
    torch.testing.assert_allclose(out, expected, rtol=1e-7, atol=0)

@pytest.mark.parametrize("input_R", [
    [0.5, 1.5, 2.5],
    np.array([0.5, 1.5, 2.5], dtype=float),
    torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32),
])
def test_accepts_various_input_types(input_R):
    """
    The function should accept list, NumPy array, and torch.Tensor, and yield consistent results.
    """
    out_list = radial_function('H', input_R, n=1, device='cpu')
    out_list2 = radial_function('H', list(input_R), n=1, device='cpu')
    torch.testing.assert_allclose(out_list, out_list2, rtol=1e-7, atol=0)

def test_gpu_device_behavior():
    """
    If CUDA is available, outputs on cuda and cpu should match (up to device).
    """
    R = [1.0, 2.0]
    cpu_out = radial_function('H', R, n=1, device='cpu')

    if torch.cuda.is_available():
        gpu_out = radial_function('H', R, n=1, device='cuda')
        assert gpu_out.device.type == 'cuda'
        torch.testing.assert_allclose(cpu_out, gpu_out.cpu(), rtol=1e-7, atol=0)
    else:
        pytest.skip("CUDA not available, skipping GPU test")

def test_zero_at_origin_for_n_gt1():
    """
    For any n > 1, at r=0: R_n = 0^(n-1) = 0
    """
    out = radial_function('He', [0.0], n=3, device='cpu')
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.0, abs=0.0)


@pytest.fixture(autouse=True)
def patch_atomic_numbers(monkeypatch):
    """
    Ensure element_atomic_numbers contains at least H=1 for our tests.
    """
    monkeypatch.setitem(element_atomic_numbers, 'H', 1)

def test_single_atom_zero_moment():
    atoms = [('H', (0.0, 0.0, 0.0))]
    M = nuclear_charge_moment(atoms, device='cpu')
    assert M.shape == (3, 3)
    torch.testing.assert_allclose(M, torch.zeros(3, 3), atol=0.0, rtol=0.0)

def test_two_hydrogens_symmetric_along_x():
    atoms = [('H', (-1.0, 0.0, 0.0)), ('H', (1.0, 0.0, 0.0))]
    M = nuclear_charge_moment(atoms, device='cpu')
    expected = torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_allclose(M, expected, rtol=1e-7, atol=0.0)

def test_unknown_element_symbol_raises_keyerror():
    atoms = [('Xx', (0.0, 0.0, 0.0))]
    with pytest.raises(KeyError):
        nuclear_charge_moment(atoms, device='cpu')

def test_bad_coordinate_length_raises_runtimeerror():
    # Passing a 2-tuple will lead to a tensor-shape mismatch at stack time.
    atoms = [('H', (0.0, 1.0))]
    with pytest.raises(RuntimeError):
        nuclear_charge_moment(atoms, device='cpu')

def test_gpu_and_cpu_consistency():
    atoms = [('H', (0.5, 0.5, 0.5)), ('H', (-0.5, -0.5, -0.5))]
    cpu_M = nuclear_charge_moment(atoms, device='cpu')
    if torch.cuda.is_available():
        gpu_M = nuclear_charge_moment(atoms, device='cuda')
        torch.testing.assert_allclose(cpu_M, gpu_M.cpu(), rtol=1e-7, atol=0.0)
    else:
        pytest.skip("CUDA unavailable, skipping GPU consistency test")