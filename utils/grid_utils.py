import numpy as np
import torch
from . import element_atomic_numbers
from .geom_utils import compute_distance_coord

def load_vxc_coordinates(file_name):
    """
    Loads VXC coordinates from a file and returns the coordinates as a torch tensor
    in double precision.

    The file is assumed to contain rows of numbers with at least four columns (x, y, z, value).
    This function reads the file, extracts the first three columns (x, y, z) from each line,
    and returns a 2D torch.Tensor where each row is an (x, y, z) coordinate in high precision.

    Parameters:
    -----------
    file_name : str
        The path to the file containing VXC coordinates.

    Returns:
    --------
    coords : torch.Tensor
        A 2D tensor (torch.float64) with each row containing the (x, y, z) coordinates.

    Raises:
    -------
    OSError
        If `file_name` cannot be opened for reading.
    """
    coords_list = []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            coords_list.append([x, y, z])
    if not coords_list:
        # ensure a (0,3) tensor when no valid lines are found
        return torch.empty((0, 3), dtype=torch.float64)
    coords = torch.tensor(coords_list, dtype=torch.float64)
    return coords

def load_density(file_name):
    """
    Extracts the density which is the fourth column from a text file and returns it as a torch tensor.

    The file is expected to have whitespace-separated values.
    
    Parameters:
    -----------
    file_name : str
        The path to the file containing the data.

    Returns:
    --------
    values : torch.Tensor
        A 1D torch tensor (torch.float64) containing the density values from the fourth column.

    Raises
    ------
    OSError
        If `file_name` cannot be opened for reading.
    """
    density_list = []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                value = float(parts[3])
                density_list.append(value)
            except ValueError:
                continue
    values = torch.tensor(density_list, dtype=torch.float64)
    return values


def cartesian_to_spherical(x, y, z, device):
    """
    Convert Cartesian coordinates to spherical coordinates using PyTorch with double precision.

    Parameters:
    -----------
    x, y, z : float, list, np.ndarray, or torch.Tensor
        Cartesian coordinates.
    device : torch.device or str
        Device on which the output tensors will be placed (e.g. "cpu" or "cuda").

    Returns:
    --------
    r : torch.Tensor
        Radial distance, r = sqrt(x² + y² + z²), dtype=torch.float64.
    θ : torch.Tensor
        Polar angle, θ = arccos(z / r), range [0, π]. If r = 0, θ = NaN.
    φ : torch.Tensor
        Azimuthal angle, φ = atan2(y, x), range (−π, π]. If (x,y) = (0,0),
        φ = 0 by convention.

    Notes
    -----
    - At the origin, r = 0 → θ is NaN (0/0) and φ = 0 (atan2(0,0)).
    - All inputs are cast to torch.float64.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64,device=device)
    else:
        x = x.double().to(device)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float64,device=device)
    else:
        y = y.double().to(device)
    if not torch.is_tensor(z):
        z = torch.tensor(z, dtype=torch.float64,device=device)
    else:
        z = z.double().to(device)

    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)

    return r, theta, phi


def real_spherical_harmonics_on_grid(l, m, x, y, z, device):
    """
    Compute the real spherical harmonics Y_l^m on a grid of Cartesian coordinates 
    and return the result as a torch tensor in double precision.

    The real spherical harmonics are constructed from SciPy’s complex sph_harm:
      • m > 0 :  Yₗᵐ = √2 · (−1)ᵐ · Im[ Yₗᵐ(complex) ]
      • m < 0 :  Yₗᵐ = √2 · (−1)ᵐ · Re[ Yₗ|m|(complex) ]
      • m = 0 :  Yₗ⁰ =    Re[ Yₗ⁰(complex) ]

    Parameters:
    -----------
    l : int
        Degree (l ≥ 0).
    m : int
        Order (−l ≤ m ≤ l).
    x, y, z : scalar, sequence, numpy.ndarray, or torch.Tensor
        Cartesian coordinates; must be broadcastable to a common shape.
    device : torch.device or str
        Device for the output tensor (e.g. 'cpu' or 'cuda').

    Returns:
    --------
    torch.Tensor
        The real spherical harmonic Y_l^m evaluated at each grid point,
        returned as a torch tensor with dtype=torch.float64.

    Raises
    ------
    RuntimeError
        If x, y, z cannot be broadcast together.
    OSError
        If SciPy’s sph_harm isn’t available.
    """
    from scipy.special import sph_harm

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64,device=device)
    else:
        x = x.double().to(device)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float64,device=device)
    else:
        y = y.double().to(device)
    if not torch.is_tensor(z):
        z = torch.tensor(z, dtype=torch.float64,device=device)
    else:
        z = z.double().to(device)

    r, theta, phi = cartesian_to_spherical(x, y, z, device)
    
    # Convert theta and phi to numpy arrays so they can be used with scipy.special.sph_harm.
    theta_np = theta.cpu().detach().numpy()
    phi_np = phi.cpu().detach().numpy()
    
    # Compute the spherical harmonic using scipy.special.sph_harm.
    if m > 0:
        sph_val = sph_harm(m, l, phi_np, theta_np)
        result_np = np.sqrt(2.0) * ((-1) ** m) * np.imag(sph_val)
    elif m < 0:
        sph_val = sph_harm(-m, l, phi_np, theta_np)
        result_np = np.sqrt(2.0) * ((-1) ** m) * np.real(sph_val)
    else:
        sph_val = sph_harm(0, l, phi_np, theta_np)
        result_np = np.real(sph_val)
    
    # Convert the resulting NumPy array to a torch tensor with high precision.
    result = torch.tensor(result_np, dtype=torch.float64,device=device)
    return result


def radial_function(atom, R, n, device):
    """
    Computes the radial part of a Slater-type orbital (STO) for a given atom and principal quantum number.

    The radial function is defined as:
        R_n(r) = r^(n-1) · exp(-ζ · r)

    where ζ is approximated using a simplified Slater rule:
        ζ ≈ (Z - 0.35(Z - 1)) / n

    Parameters
    ----------
    atom : str
        Chemical symbol of the atom (e.g., "C" for carbon).
    R : array-like or torch.Tensor
        Radial grid points (assumed to be in atomic units, i.e., bohr).
    n : int
        Principal quantum number.
    device : torch.device or str
        Device on which to perform the computation (e.g., "cuda" or "cpu").

    Returns
    -------
    torch.Tensor
        Radial function values R_n(r) evaluated at each point in `R`, on the specified device.

    Notes
    -----
    - This function does not include normalization.
    - The input `R` can be any shape; the result will match it.
    - Atomic numbers are retrieved from a global dictionary `element_atomic_numbers`.
    - Assumes atomic units throughout.
    """
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float64).to(device)
    else:
        R = R.double()

    Z = element_atomic_numbers[atom]
    zeta_val = (Z - 0.35 * (Z - 1)) / n
    zeta = torch.as_tensor(float(zeta_val), dtype=R.dtype, device=device)

    R = R.to(device)

    R_n = R.pow(n - 1) * torch.exp(-zeta * R)

    return R_n


def nuclear_charge_moment(atoms, device):
    """
    Compute the second moment (quadrupole-like tensor) of the nuclear charge distribution.

    Given a list of atoms with their symbols and Cartesian coordinates, this function
    constructs the 3×3 tensor:
        M = Σ_A Z_A [ |r_A − T|² I₃ − (r_A − T) ⊗ (r_A − T) ]
    where:
      - Z_A is the atomic number of atom A,
      - r_A is the position vector of atom A,
      - T is the center of charge: T = (Σ_A Z_A r_A) / (Σ_A Z_A),
      - I₃ is the 3×3 identity matrix,
      - ⊗ denotes the outer product.

    Args:
        atoms (List[Tuple[str, Tuple[float, float, float]]]):
            A sequence of tuples, each containing:
              - element symbol (str), e.g. "C", "H", "O"
              - coordinates as a length‑3 tuple of floats (x, y, z).
        device (torch.device or str):
            The torch device on which to allocate tensors (e.g., 'cpu' or 'cuda:0').

    Returns:
        torch.Tensor:
            A 3×3 tensor of dtype torch.float64 on the specified device, representing
            the nuclear charge moment tensor M.

    Raises:
        KeyError:
            If an element symbol in `atoms` is not found in `element_atomic_numbers`.
        ValueError:
            If any coordinate tuple is not of length 3 or cannot be cast to float.
    """
    # --- 1. build tensors for positions and charges ---
    coords = []
    charges = []
    for sym, c in atoms:
        Z = element_atomic_numbers[sym]
        charges.append(Z)
        t = torch.as_tensor(c, dtype=torch.float64, device=device)
        coords.append(t)
    coords = torch.stack(coords, dim=0)
    charges = torch.tensor(charges, dtype=torch.float64, device=device)

    # --- 2. compute center of charge T ---
    total_Z = charges.sum()
    T = (charges[:,None] * coords).sum(dim=0) / total_Z
    
    # --- 3. displacements r_A ---
    r = coords - T
    r2 = (r**2).sum(dim=1)

    # --- 4. build each term and sum ---
    I3 = torch.eye(3, dtype=torch.float64, device=device)
    # outer products: shape (N,3,3)
    outer = torch.einsum('ni,nj->nij', r, r)
    # term per atom: [r2 I - outer]
    terms = r2[:,None,None] * I3 - outer          # (N,3,3)
    M = (charges[:,None,None] * terms).sum(dim=0) # (3,3)
    return M
    
def diagonalize_moment_tensor(M, tol=1e-8):
    """
    Given a real symmetric moment tensor M (shape [3,3]), compute
        M = O @ Λ @ O.T
    with O^T O = I and Λ diagonal.

    Returns:
    --------
    O      : torch.Tensor, shape (3,3)
             Orthonormal eigenvector matrix (columns are eigenvectors).

    Lambda : torch.Tensor, shape (3,3)
             Diagonal matrix of eigenvalues.

    Raises:
    -------
    AssertionError if either sanity check fails beyond `tol`.
    """
    eigenvals, eigenvecs = torch.linalg.eigh(M)
    # eigenvals: sorted in ascending order
    # eigenvecs: columns are the unit–norm eigenvectors
    O = eigenvecs                 # (3,3)
    Lambda = torch.diag(eigenvals)     # (3,3)

    # sanity check 1: O^T M O == Lambda
    res1 = O.T @ M @ O
    if not torch.allclose(res1, Lambda, atol=tol, rtol=0):
        max_err = (res1 - Lambda).abs().max()
        raise AssertionError(f"O^T M O != Λ (max abs error {max_err:.3e})")
    
    # sanity check 2: O^T O == I
    I3   = torch.eye(M.size(0), dtype=M.dtype, device=M.device)
    res2 = O.T @ O
    if not torch.allclose(res2, I3, atol=tol, rtol=0):
        max_err = (res2 - I3).abs().max()
        raise AssertionError(f"O^T O != I (max abs error {max_err:.3e})")
    
    return O, Lambda


def rotate_grid(XYZ, O, device):
    """
    Apply a 3×3 rotation matrix to a set of 3D points.

    This function multiplies each row in the input tensor `XYZ` by the rotation
    matrix `O` to produce rotated coordinates.

    Args:
        XYZ (torch.Tensor): A 2D tensor of shape (N, 3), where N is the number
            of points and each row represents a 3D coordinate.
        O (torch.Tensor): A 2D tensor of shape (3, 3) representing a rotation
            matrix. Must be orthogonal with determinant +1 for a proper rotation.

    Returns:
        torch.Tensor:
            A 2D tensor of shape (N, 3) containing the rotated coordinates,
            with the same dtype and device as `XYZ`.

    Raises:
        AssertionError: If `XYZ` is not a 2D tensor with size[1] == 3, or if
            `O` does not have shape (3, 3).
    """
    assert XYZ.dim() == 2 and XYZ.size(1) == 3
    assert O.shape == (3, 3)

    # x' = x @ O
    XYZ = XYZ.to(device)
    return XYZ.matmul(O)
    

def extract_nlm(indices):
    """
    Extracts the quantum numbers n, l, and m from a list of indices.

    Parameters:
    -----------
    indices : list of str
        A list containing the quantum number characters.

    Returns:
    --------
    tuple
        A tuple (n, l, m) where each element is either an integer or None if the value
        could not be extracted.
    """
    n = int(indices[0]) if len(indices) > 0 and indices[0].isdigit() else None
    l = int(indices[1]) if len(indices) > 1 and indices[1].isdigit() else None
    m = None

    if len(indices) > 2:
        if indices[2] == '-' and len(indices) > 3 and indices[3].isdigit():
            m = -int(indices[3])
        elif indices[2].isdigit():
            m = int(indices[2])

    return n,l,m

def bsf(a0,a1,a2):
    """
    Computes a linear combination of its arguments.
    All inputs are cast to torch tensors in double precision.
    """
    a0 = torch.as_tensor(a0, dtype=torch.float64)
    a1 = torch.as_tensor(a1, dtype=torch.float64)
    a2 = torch.as_tensor(a2, dtype=torch.float64)
    return 0.35 * a0 + 0.85 * a1 + a2

def get_radii_2(Z):
    """
    Computes a radius based on the atomic number Z.
    Z is assumed to be an integer (Python scalar).
    Returns a torch scalar in double precision.
    """
    if Z>10:
        print(" Becke partitioning not ready for atomic number higher than 10 \n")
        return None
    
    zn = [2, 10]
    zv = [1, 2]

    for i in range(2):
        if Z <= zn[i]:
            val0 = zv[i]
            break

    if Z==1:
        val1 = 0.0
    elif Z==2:
        val1 = 0.3
    elif Z<=10:
        val1 = bsf(Z-3,2.0,0.0)

    Z_tensor = torch.tensor(Z, dtype=torch.float64)
    numerator = torch.tensor(val0 * val0, dtype=torch.float64)
    r1 = numerator / (Z_tensor - val1)

    if Z==3:
        r1 = torch.tensor(2.0, dtype=torch.float64)

    return r1


def becke_a(Z1,Z2,alpha=1):
    """
    Computes the Becke atomic partitioning parameter.
    Z1, Z2 are atomic numbers (Python integers).
    Returns a torch tensor (a scalar) in double precision.
    """
    r1 = get_radii_2(Z1)
    r2 = get_radii_2(Z2)
    x = alpha * r1 / r2
    u = (x-1)/(x+1)
    a = u/(u*u-1)

    a = torch.clamp(a, -0.5, 0.5)
    return a

def bf3(mu):
    """
    Applies a three-step iterative function on f1.
    f1 can be a torch tensor (possibly with more than one element).
    """
    # calculates series of functions f_k(x) = f_1(f_k-1)
    # where f_k(x) = 1.5x - 0.5x^3
    # standard is to stop at f_3 order
    # finally returns the switching function
    # s_AB(r) = 0.5(1-f_3(nu_AB(r)))
    # for _ in range(3):
    #     mu = 1.5 * mu - 0.5 * (mu**3)
    mu = torch.clamp(mu, -1.0 + 1e-15, 1.0 - 1e-15)
    for _ in range(3):                       # g_{k+1}(x) = 1.5 x - 0.5 x^3
        mu = 1.5 * mu - 0.5 * mu**3

    # return 0.5 * (1.0 - mu)  
    mu = 0.5*(1-mu)
    mu = mu**2
    return mu
    

def becke_weight(coords, charges, XYZ, device):
    natoms = len(coords)
    if not isinstance(XYZ, torch.Tensor):
        XYZ = torch.tensor(XYZ, dtype=torch.float64, device = device)
    else:
        XYZ = XYZ.double().to(device)

    gsa = XYZ.shape[0]

    # Distances[i, A] = distance from grid point i to atom A
    # shape: (num_points, num_atoms)
    distances = torch.zeros((gsa, natoms), dtype=torch.float64,device=device)
    for A in range(natoms):
        diff = XYZ - coords[A]
        distances[:, A] = torch.sqrt(torch.sum(diff**2, dim=1))

    partial_wts = torch.ones((gsa, natoms), dtype=torch.float64, device=device)

    for A in range(natoms):
        for B in range(natoms):
            if B==A:
                continue
            rA = distances[:,A]
            rB = distances[:,B]
            R_AB = compute_distance_coord(coords[A],coords[B])
            R_AB = R_AB.clamp_min_(1e-12)

            mu_AB = (rA - rB)/R_AB

            Z_A = charges[A]
            Z_B = charges[B]
            
            a_AB = becke_a(Z_A, Z_B)
            nu_AB = mu_AB + a_AB * (1 - mu_AB**2)
            s_AB = bf3(nu_AB)
            
            partial_wts[:, A] *= s_AB

    # Normalize: w_A(r_i) = f_A(r_i) / sum_{C} f_C(r_i)
    denom = torch.sum(partial_wts, dim=1).clamp_min(1e-15)
    weights = partial_wts / denom[:, None]
    
    return weights


def make_basis(atom,XYZ,n,l,m,atom_coords,atom_wt,device):
    """
    Computes the basis function for a given atom, quantum numbers, and coordinates.
    This version uses high-precision Torch tensors.

    Parameters:
    -----------
    atom : str
        The atom symbol for which the basis function is calculated.
    XYZ : array-like or torch.Tensor
        A 2D array/tensor of shape (N, 3) containing the Cartesian coordinates of grid points.
    n : int
        The principal quantum number.
    l : int
        The orbital angular momentum quantum number.
    m : int
        The magnetic quantum number.
    atom_coords : array-like or torch.Tensor
        A 1D array/tensor of shape (3,) containing the atom's (X, Y, Z) coordinates.
    atom_wt : float or array-like or torch.Tensor
        A scalar or tensor weight for the atom.

    Returns:
    --------
    torch.Tensor
        The computed basis function evaluated at each grid point (as a high-precision tensor).
    """
    if not torch.is_tensor(XYZ):
        XYZ = torch.tensor(XYZ, dtype=torch.float64)
    else:
        XYZ = XYZ.double()

    if not torch.is_tensor(atom_coords):
        atom_coords = torch.tensor(atom_coords, dtype=torch.float64)
    else:
        atom_coords = atom_coords.double()

    if not torch.is_tensor(atom_wt):
        atom_wt = torch.tensor(atom_wt, dtype=torch.float64)
    else:
        atom_wt = atom_wt.double()

    XYZ_shifted = XYZ - atom_coords
    X_shifted = XYZ_shifted[:, 0]
    Y_shifted = XYZ_shifted[:, 1]
    Z_shifted = XYZ_shifted[:, 2]

    R = torch.sqrt(X_shifted**2 + Y_shifted**2 + Z_shifted**2)

    radial = radial_function(atom, R, n, device)
    harmonics = real_spherical_harmonics_on_grid(l,m,X_shifted,Y_shifted,Z_shifted)

    return radial * harmonics * atom_wt


def Psi_matrix(atom,XYZ,node_coords,atom_wt,device,nlm=['100','200','211','210','21-1']):
    """
    Computes a Psi matrix by stacking basis functions computed over a set of quantum numbers.
    All inputs are processed to use high precision (torch.float64).

    Parameters:
    -----------
    atom : str
        Atom symbol.
    XYZ : array-like or torch.Tensor
        Grid points (N x 3) in Cartesian coordinates.
    node_coords : array-like or torch.Tensor
        The atom's coordinates (3,).
    atom_wt : float or array-like or torch.Tensor
        The atomic weight.
    device : torch.device
        The device on which to place the tensors.
    nlm : list of str, optional
        A list of strings encoding the quantum numbers, e.g. ['100', '200', '211', ...].

    Returns:
    --------
    torch.Tensor
        A Psi matrix tensor of shape (N, number_of_functions), in double precision.
    """
    basis_tensors = []
    expected_shape = None
    for indices in nlm:
        n, l, m = extract_nlm(indices)
        basis = make_basis(atom, XYZ, n, l, m, node_coords, atom_wt, device)
        # basis = basis.to(device)
        if expected_shape is None:
            expected_shape = basis.shape
        elif basis.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {basis.shape}")
        basis_tensors.append(basis)
    Psi = torch.stack(basis_tensors, dim=1)
    return Psi