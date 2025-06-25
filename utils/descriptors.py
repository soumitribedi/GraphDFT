
import numpy as np
import os
import torch
from . import element_atomic_numbers
from .geom_utils import read_geom_file, read_descriptor_file
from .grid_utils import load_vxc_coordinates, load_density, Psi_matrix, extract_nlm, becke_weight, nuclear_charge_moment, diagonalize_moment_tensor, rotate_grid
from .overlap import get_overlap_mol, orthogonalize_basis


def project_density_integrate(psi_tilde, rho, w):
    """
    Computes the expansion coefficients c_i.
    
    Parameters:
    - psi_tilde (torch.Tensor): Orthogonalized basis functions (M x N).
    - rho (torch.Tensor): Electron density (M x 1).
    - w (torch.Tensor): Integration weights (M x 1).
    
    Returns:
    - c (torch.Tensor): Expansion coefficients (N x 1).
    """
    # psi_tilde.conj(): (M x N), rho: (M x 1), w: (M x 1)
    # integrand: (M x N)
    integrand = torch.einsum('ni,nj,nj->ni',psi_tilde, rho, w)
    # Sum over M (grid points), resulting in (N,)
    c = torch.sum(integrand, dim=0, keepdim=True).T  # Transpose to get (N x 1)
    return c


def create_descriptors_molecule(mol_dir,device,nlm=['100','200','211','210','21-1'],rho=None):
    geom_file = os.path.join(mol_dir,"GEOM")
    if rho is None:
        density_file = os.path.join(mol_dir,"rho_wf")
        rho = load_density(density_file)
    atoms = read_geom_file(geom_file)
    XYZ_orig = load_vxc_coordinates(os.path.join(mol_dir,"gridwts"))
    wt = load_density(os.path.join(mol_dir,"gridwts"))

    if not torch.is_tensor(rho):
        rho = torch.tensor(rho, dtype=torch.float64)
    else:
        rho = rho.double()
    if not torch.is_tensor(wt):
        wt = torch.tensor(wt, dtype=torch.float64)
    else:
        wt = wt.double()
    rho = rho.to(device).view(-1, 1)
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

    descriptor_dict = {}
    Psi_list = []
    keys_list = []
    for atom_index, (element, _) in enumerate(atoms):
        coord = coords[atom_index]
        Z = charges[atom_index]
        if Z is None:
            raise ValueError(f"Unknown element symbol: {element}")

        # Use a unique key for each atom, e.g., atom index or coordinates
        atom_key = (element, atom_index, Z)
        atom_wt = partial_weights[:,atom_index]
        Psi_atomic = Psi_matrix(element,XYZ,coord,atom_wt,device,nlm)
        Psi_list.append(Psi_atomic)

        # Collect keys for each basis function
        for indices in nlm:
            if isinstance(indices, list):
                indices_key = tuple(indices)
            else:
                indices_key = indices  # Assuming it's already hashable

            # Append to keys_list
            keys_list.append((atom_key, indices_key))

    Psi_tensor = torch.cat(Psi_list,dim=1)

    overlap_file = os.path.join(mol_dir,"S_tensor.pt")
    if os.path.exists(overlap_file):
        S = torch.load(overlap_file, map_location=device)
    else:
        S = get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device, nlm,write=True)

    Psi_tensor_tilde = orthogonalize_basis(Psi_tensor, S, wt)

    descriptor = project_density_integrate(Psi_tensor_tilde, rho, wt)

    descriptor_dict = {}
    for (atom_key, indices_key), coef in zip(keys_list, descriptor):
        descriptor_dict.setdefault(atom_key, {})[indices_key] = coef

    return descriptor_dict


def convert_descriptors_to_format(descriptor_dict, nlm=['100','200','211','210','21-1']):
    output_entries = []

    for atom_key, indices_dict in descriptor_dict.items():
        element, _, Z = atom_key
        entry = [element, Z]

        for indices in nlm:
            descriptor_value = indices_dict.get(indices, 'NA')
            if descriptor_value != 'NA':
                descriptor_value = descriptor_value.item()  # precise float64 conversion
            entry.append(descriptor_value)

        output_entries.append(entry)
    return output_entries

def save_descriptors_to_csv(descriptors_list, filename, nlm=['100', '200', '211', '210', '21-1']):
    import csv

    headers = ['Element', 'Z'] + ['C' + indices for indices in nlm]

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)  # Write the headers
        csvwriter.writerows(descriptors_list)  # Write the data rows
    return

def create_descriptor(device,nlm=['100','200','211','210','21-1'],dataset_file='dataset_small.csv', 
                 geom_folder='dataset',descriptor_file_name='descriptors.csv',rho=None):
    import csv
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            descriptor_file = os.path.join(mol_dir,descriptor_file_name)
            if not os.path.exists(descriptor_file):
                descriptors_dict = create_descriptors_molecule(mol_dir,device,nlm,rho)
                descriptors_list = convert_descriptors_to_format(descriptors_dict,nlm)
                save_descriptors_to_csv(descriptors_list,descriptor_file,nlm)
                print(f"Creating descriptors {descriptor_file_name} in {mol_dir}")
            else:
                print(f"File already exists. Skipping save.")
    return


def delete_descriptor(dataset_file='dataset_small.csv', 
                 geom_folder='dataset',descriptor_file = 'descriptors.csv',verbose=1):
    import csv
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            descriptor_file_path = os.path.join(mol_dir,descriptor_file)
            if os.path.exists(descriptor_file_path):
                os.remove(descriptor_file_path)
                if verbose>0:
                    print(f"Deleted descriptor file {descriptor_file} in {mol_dir}")
            else:
                if verbose>0:
                    print("Descriptor file does not exist. Skipping delete")
    return