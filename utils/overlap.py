import os
import csv
import torch
from . import element_atomic_numbers
from .grid_utils import load_vxc_coordinates, Psi_matrix, nuclear_charge_moment, diagonalize_moment_tensor, rotate_grid, becke_weight, load_density
from .geom_utils import read_geom_file


def compute_overlap_matrix(psi, w):
    """
    Compute the overlap matrix S given basis-function values and quadrature weights.

    The overlap matrix is defined as:
        S_{ij} = ∫ ψ_i*(r) ψ_j(r) dr
    and is approximated by discrete quadrature:
        S = (ψ_conj * w)ᵀ @ ψ

    This version accepts `w` either as a 1‑D tensor of shape (M,) or as a
    2‑D column vector of shape (M,1).  Internally it always reshapes to (M,1)
    so that each row of ψ is weighted by the corresponding quadrature weight.

    Args:
        psi (torch.Tensor): shape (M, N), values of N basis functions on M points.
                           May be real or complex.
        w   (torch.Tensor): shape (M,) or (M,1), quadrature weights for each of
                           the M grid points.

    Returns:
        torch.Tensor: the (N, N) overlap matrix S, same dtype (and device) as ψ.

    Raises:
        ValueError: if `w` cannot be interpreted as a (M,) or (M,1) vector matching
                    the first dimension of `psi`.
    """
    # psi: (M x N), w: (M x 1), psi_conj_weighted: (M x N)
    psi_conj_weighted = psi.conj() * w
    S = psi_conj_weighted.T @ psi  # Resulting S: (N x N)
    # no need to normalize since the weights integrate to correct value
    return S


def get_overlap_mol(mol_dir,atoms,XYZ,coords,partial_weights,gridwts,device,nlm=['100','200','211','210','21-1'],write=True):
    """
    Compute and optionally save the overlap matrix for a molecule using atom-centered basis functions.

    For each atom in `atoms`, this function evaluates its basis functions (specified
    by `nlm`) on the grid `XYZ` using `Psi_matrix`, weights them by `partial_weights`,
    and concatenates the results into a global basis matrix. It then computes the
    overlap matrix S = Ψ† W Ψ via discrete quadrature and returns it. If `write=True`,
    the resulting tensor is saved to `<mol_dir>/S_tensor.pt`.

    Args:
        mol_dir (str):
            Directory path where the overlap tensor will be saved if `write=True`.
        atoms (List[Tuple[str, Tuple[float, float, float]]]):
            List of (element_symbol, coordinate) tuples describing each atom.
        XYZ (torch.Tensor):
            A (n, 3) tensor of grid point coordinates on which basis functions are evaluated.
        coords (Sequence[Sequence[float]]):
            Sequence of length‑3 coordinate tuples or tensors for each atom, matching `atoms`.
        partial_weights (torch.Tensor):
            A (n, natoms) tensor of partial quadrature weights for each grid point and atom.
        device (torch.device or str):
            The torch device on which computations and the saved tensor will reside.
        nlm (List[str], optional):
            List of basis-function labels (e.g., angular quantum numbers) to evaluate per atom.
            Defaults to ['100','200','211','210','21-1'].
        write (bool, optional):
            If True, save the computed overlap matrix S to `<mol_dir>/S_tensor.pt`.
            Defaults to True.

    Returns:
        torch.Tensor:
            The (N_basis, N_basis) overlap matrix S of dtype torch.float64 on `device`,
            where N_basis = natoms * len(nlm).

    Raises:
        RuntimeError:
            If any underlying tensor operation fails (e.g., shape mismatches in Psi_matrix).
        FileNotFoundError:
            If `mol_dir` does not exist or is not writable when `write=True`.
    """
    Psi_list = []
    for i, atom in enumerate(atoms):
        element, _ = atom
        coords_at = coords[i]
        atom_wt = partial_weights[:,i]
        Psi_atomic = Psi_matrix(element,XYZ,coords_at,atom_wt,device,nlm)
        Psi_list.append(Psi_atomic)
    Psi_tensor = torch.cat(Psi_list,dim=1)
    Psi_tensor = Psi_tensor.to(device)
    S = compute_overlap_matrix(Psi_tensor, gridwts)
    if write==True:
       torch.save(S, os.path.join(mol_dir,"S_tensor.pt"))
    return S


def create_overlap(device,nlm=['100','200','211','210','21-1'],dataset_file='dataset_small.csv', geom_folder='dataset'):
    """
    Iterate over a molecular dataset and compute/write overlap matrices where missing.

    For each molecule listed in `<geom_folder>/<dataset_file>`, this function:
      1. Checks for an existing `<mol_dir>/S_tensor.pt`.  
      2. If missing:
         a. Reads atomic geometry from `<mol_dir>/GEOM`.  
         b. Loads grid coordinates and weights from `<mol_dir>/gridwts`.  
         c. Computes the nuclear charge moment tensor and its principal axes.  
         d. Rotates the grid and atomic coordinates to align with the principal axes.  
         e. Computes Becke partition weights on the rotated grid.  
         f. Calls `get_overlap_mol` to assemble and save the overlap matrix.  
         g. Prints a status message.  
      3. If present, skips that molecule and prints a skip message.

    Args:
        device (torch.device or str):
            Device on which all tensors (grid, coordinates, charges) will be allocated
            (e.g., 'cpu' or 'cuda:0').
        nlm (List[str], optional):
            List of basis-function labels (quantum numbers) to use for each atom.
            Defaults to ['100','200','211','210','21-1'].
        dataset_file (str, optional):
            Filename (CSV) listing molecule IDs under `geom_folder`. Each row
            should start with the molecule ID. Defaults to 'dataset_small.csv'.
        geom_folder (str, optional):
            Path to the top‐level directory containing per‐molecule subfolders.
            Defaults to 'dataset'.

    Returns:
        None

    Raises:
        FileNotFoundError:
            If `dataset_file` or any expected per‐molecule file (GEOM, gridwts) is missing.
        KeyError:
            If an element symbol in a GEOM file is not found in `element_atomic_numbers`.
        RuntimeError:
            If any of the underlying tensor operations (e.g., moment diagonalization,
            Becke weights, overlap assembly) fails.
    """
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            if not os.path.exists(os.path.join(mol_dir,"S_tensor.pt")):
                geom_file = os.path.join(mol_dir,"GEOM")
                atoms = read_geom_file(geom_file)
                XYZ_orig = load_vxc_coordinates(os.path.join(mol_dir,"gridwts"))
                wt = load_density(os.path.join(mol_dir,"gridwts"))
                if not torch.is_tensor(wt):
                    wt = torch.tensor(wt, dtype=torch.float64)
                else:
                    wt = wt.double()
                wt = wt.to(device).view(-1, 1) # Shape: (581000, 1)
                M = nuclear_charge_moment(atoms, device)
                O, _ = diagonalize_moment_tensor(M)
                XYZ = rotate_grid(XYZ_orig, O, device)
                coords_orig = []
                charges = []
                for sym, c in atoms:
                    Z = element_atomic_numbers[sym]
                    charges.append(Z)
                    t = torch.as_tensor(c, dtype=torch.float64, device=device)
                    coords_orig.append(t)
                coords_orig = torch.stack(coords_orig, dim=0)

                charges = torch.tensor(charges, dtype=torch.float64, device=device)
                coords = rotate_grid(coords_orig, O, device)

                partial_weights = becke_weight(coords, charges, XYZ,device)
                get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device, nlm,write=True)
                print(f"Writing overlap file in {mol_dir}")
            else:
                print(f"File already exists. Skipping save.")
    return

def load_overlap(batch,device,verbose=0):
    num_graphs = batch.ptr.size(0) - 1
    S_list = []
    for graph_idx in range(num_graphs):
        molecule_name = batch.name[graph_idx]
        if verbose>0:
            print(f"Loading overlap matrix for {molecule_name}")
        S_file = os.path.join("dataset",molecule_name,"S_tensor.pt")
        S_mol = torch.load(S_file, map_location=device)
        S_list.append(S_mol)
    S_tensor = torch.cat(S_list,dim=0)
    return S_tensor


def delete_overlap(dataset_file='dataset_small.csv', 
                 geom_folder='dataset'):
    import csv
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            if os.path.exists(os.path.join(mol_dir,"S_tensor.pt")):
                os.remove(os.path.join(mol_dir,"S_tensor.pt"))
                print(f"Deleted overlap file S_tensor.pt in {mol_dir}")
            else:
                print("Overlap file does not exist. Skipping delete")
    return


def orthogonalize_basis(Psi_tensor, S, w):
    """
    Computes an orthogonalized basis from Psi_tensor using the overlap matrix S and integration weights w.
    
    Parameters:
    - Psi_tensor (torch.Tensor): Basis functions (M x N).
    - S (torch.Tensor): Overlap matrix (N x N).
    - w (torch.Tensor): Integration weights (M x 1).
    
    Returns:
    - Psi_tilde (torch.Tensor): Orthogonalized basis functions (M x N) in high precision.
    """
    D, U = torch.linalg.eigh(S)
    # Invert the square roots of the eigenvalues
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    # Compute S^{-1/2}
    S_inv_sqrt = U @ D_inv_sqrt @ U.conj().T
    # Orthogonalize the basis functions
    # psi: (M x N), psi_tilde: (M x N)
    Psi_tilde = Psi_tensor @ S_inv_sqrt

    # Normalize each basis function
    norms_squared = torch.sum((Psi_tilde.conj() * Psi_tilde) * w, dim=0).real
    # Add a small epsilon to prevent division by zero
    epsilon = torch.tensor(1e-15, dtype=Psi_tensor.dtype)
    norms = torch.sqrt(norms_squared + epsilon)
    Psi_tilde = Psi_tilde / norms
    return Psi_tilde

def print_overlap(S):
    S_np = S.cpu().numpy()

    # Create a DataFrame
    import pandas as pd
    df = pd.DataFrame(S_np)

    # Optionally, set display options
    pd.set_option('display.precision', 10)

    # Print the DataFrame
    print(df)
    return

def verify_orthonormality(psi_tilde, w, tol=1e-8, plot=True):
    """
    Verifies the orthonormality of the orthogonalized basis functions psi_tilde.

    Parameters:
    - psi_tilde (torch.Tensor): Orthogonalized basis functions, shape (M, N)
    - w (torch.Tensor): Integration weights, shape (M, 1)
    - tol (float): Tolerance for orthonormality check

    Returns:
    - max_deviation (float): Maximum absolute deviation from orthonormality
    - is_orthonormal (bool): True if the basis functions are orthonormal within the given tolerance
    """
    # Ensure psi_tilde and w are on the same device
    device = psi_tilde.device
    w = w.to(device)

    # Compute the overlap matrix of the orthogonalized basis functions
    # psi_tilde: (M, N), w: (M, 1)
    # psi_tilde_conj_weighted: (M, N)

    psi_tilde_conj_weighted = psi_tilde.conj() * w  # Element-wise multiplication
    S_tilde = psi_tilde_conj_weighted.T @ psi_tilde  # Shape: (N, N)

    # Create identity matrix of appropriate size and type
    N = S_tilde.shape[0]
    I = torch.eye(N, dtype=S_tilde.dtype, device=device)

    # Compute the deviation from the identity matrix
    deviation = S_tilde - I

    # Compute the maximum absolute deviation
    max_deviation = torch.max(torch.abs(deviation)).item()

    # Check if the maximum deviation is within the tolerance
    is_orthonormal = max_deviation < tol

    # Print the result
    if is_orthonormal:
        print(f"The basis functions are orthonormal within the tolerance of {tol}.")
    else:
        print(f"The basis functions are NOT orthonormal within the tolerance of {tol}.")
        print(f"Maximum absolute deviation from orthonormality: {max_deviation}")

    print_overlap(S_tilde)

    if plot:
        import matplotlib.pyplot as plt

        abs_S_tilde = torch.abs(S_tilde).cpu().numpy()
        plt.figure(figsize=(6, 5))
        plt.imshow(abs_S_tilde, cmap='viridis')
        plt.colorbar()
        plt.title('Absolute Value of Overlap Matrix S_tilde')
        plt.xlabel('Basis Function Index')
        plt.ylabel('Basis Function Index')
        plt.show()

    return max_deviation, is_orthonormal