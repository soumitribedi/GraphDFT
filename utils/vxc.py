import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from . import element_atomic_numbers
from .geom_utils import read_geom_file, read_descriptor_file
from .grid_utils import load_vxc_coordinates, load_density, Psi_matrix, becke_weight, nuclear_charge_moment, diagonalize_moment_tensor, rotate_grid
from .overlap import get_overlap_mol, orthogonalize_basis, verify_orthonormality

def plot_vxc(vxc,element,grid_coords,fig_dir,graph_idx):
    grid_coords = np.array(grid_coords, dtype=float)

    # Filter coordinates where x and y are near zero
    mask = np.abs(grid_coords[:, 0]) < 1e-8
    mask &= np.abs(grid_coords[:, 1]) < 1e-8

    Z = grid_coords[mask, 2]
    vxc_tensor = torch.tensor(vxc)
    vxc_array = vxc_tensor.cpu().numpy()
    V = vxc_array[mask]

    sorted_indices = np.argsort(Z)
    Z = Z[sorted_indices]
    V = V[sorted_indices]
    
    plt.plot(Z, V, linestyle='solid', markersize=2, marker='o', label=element)
    plt.legend()
    if fig_dir!=None:
        plt.savefig(fig_dir + "/Vxc_against_Z_Fig" + element + ".png", dpi=500)
    plt.show()
    plt.savefig(fig_dir + f"/Vxc_against_Z_Fig_{graph_idx}.png", dpi=500)


def calc_vxc_molecule(grads_graph,molecule_name,device,nlm=['100','200','211','210','21-1'],geom_folder="dataset",verbose=0):
    mol_dir = os.path.join(geom_folder,molecule_name)
    wt_file = os.path.join(mol_dir,"gridwts")
    geom_file = os.path.join(mol_dir,"GEOM")
    atoms = read_geom_file(geom_file)
    XYZ_orig = load_vxc_coordinates(wt_file)
    wt = load_density(wt_file)
    if not torch.is_tensor(wt):
        wt = torch.tensor(wt, dtype=torch.float64)
    else:
        wt = wt.double()
    wt = wt.to(device).view(-1, 1) # Shape: (581000, 1)

    M = nuclear_charge_moment(atoms, device)
    O, _ = diagonalize_moment_tensor(M)
    XYZ = rotate_grid(XYZ_orig, O, device)
    
    coords = []
    charges = []
    for sym, c in atoms:
        Z = element_atomic_numbers[sym]
        charges.append(Z)
        t = torch.as_tensor(c, dtype=torch.float64, device=device)
        coords.append(t)
    coords_orig = torch.stack(coords, dim=0)
    charges = torch.tensor(charges, dtype=torch.float64, device=device)
    coords = rotate_grid(coords_orig, O, device)

    partial_weights = becke_weight(coords, charges, XYZ,device)

    if verbose>0:
        print(f"Length of XYZ of molecule: {len(XYZ)}")

    Psi_list = []
    for i,atom in enumerate(atoms):
        element, _ = atom
        coord_at = coords[i]
        atom_wt = partial_weights[:,i]
        Psi_atomic = Psi_matrix(element,XYZ,coord_at,atom_wt,device,nlm)
        Psi_list.append(Psi_atomic)
    Psi_tensor = torch.cat(Psi_list,dim=1)
    Psi_tensor = Psi_tensor.to(device)

    if Psi_tensor.dtype != grads_graph.dtype:
        grads_graph = grads_graph.to(Psi_tensor.dtype) 

    overlap_file = os.path.join(mol_dir,"S_tensor.pt")
    if os.path.exists(overlap_file):
        S = torch.load(overlap_file, map_location=device)
    else:
        S = get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device,nlm, write=True)

    Psi_tensor_tilde = orthogonalize_basis(Psi_tensor, S, wt)
    
    vxc_pred_r = torch.sum(Psi_tensor_tilde * grads_graph, dim=1).to(device)
    return vxc_pred_r


def pred_vxc(model,batch,device,fig_dir,nlm=['100','200','211','210','21-1'],geom_folder="dataset",plot=False,verbose=0):
    model.eval()

    batch = batch.to(device)
    coords_batch = batch.coords

    batch.x.requires_grad = True
    # Clear any previously computed gradients
    model.zero_grad()

    pred, _ = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
    pred.backward(torch.ones_like(pred))
    
    # Store gradients and move to device
    C_vxc_pred_batch = torch.autograd.grad(pred, batch.x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    num_graphs = batch.ptr.size(0) - 1  # Number of graphs in the batch
    batch_vxc = []
    # Iterate over each graph in the batch
    for graph_idx in range(num_graphs):
        molecule_name = batch.name[graph_idx]
        start_idx = batch.ptr[graph_idx].item()
        end_idx = batch.ptr[graph_idx + 1].item()
    
        # Extract gradients for the current graph
        C_vxc_pred_mol = C_vxc_pred_batch[start_idx:end_idx]
    
        if verbose>0:
            coords_graph = coords_batch[start_idx:end_idx]
            print(f"Processing molecule: {molecule_name}")
            print(coords_graph)
        vxc_mol = calc_vxc_molecule(C_vxc_pred_mol, molecule_name, device, nlm, geom_folder,verbose)
        batch_vxc.append(vxc_mol)
        XYZ = load_vxc_coordinates(os.path.join(geom_folder,molecule_name,"vxc"))
        if plot==True:
            plot_vxc(vxc_mol,molecule_name,XYZ,fig_dir,graph_idx)
    return batch_vxc


def real_vxc(batch,device,geom_folder='dataset'):
    num_graphs = batch.ptr.size(0) - 1
    batch_vxc = []
    for graph_idx in range(num_graphs):
        molecule_name = batch.name[graph_idx]
        file_name = os.path.join(geom_folder,molecule_name,"vxc")
        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = np.loadtxt(lines, usecols=(0,1,2,3))
        vxc = data[:,3]
        batch_vxc.append(torch.tensor(vxc, device=device))
    return torch.cat(batch_vxc)


def get_coeff(mol_dir,device,nlm=['100','200','211','210','21-1'],verify=False,write=True,verbose=0):
    geom_file = os.path.join(mol_dir,"GEOM")
    vxc_file = os.path.join(mol_dir,"vxc")
    vxc_real = load_density(vxc_file)
    XYZ_orig = load_vxc_coordinates(vxc_file)
    atoms = read_geom_file(geom_file)
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
    Psi_list = []
    for i,atom in enumerate(atoms):
        element, _ = atom
        coords_at = coords[i]
        atom_wt = partial_weights[:,i]
        Psi_atomic = Psi_matrix(element,XYZ,coords_at,atom_wt,device,nlm)
        Psi_list.append(Psi_atomic)
    Psi_tensor = torch.cat(Psi_list,dim=1)
    Psi_tensor = Psi_tensor.to(device)

    overlap_file = os.path.join(mol_dir,"S_tensor.pt")
    if os.path.exists(overlap_file):
        if verbose>0:
            print("Reading overlap matrix from existing file")
        S = torch.load(overlap_file, map_location=device)
    else:
        S = get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device, nlm,write=True)

    if not torch.is_tensor(vxc_real):
        vxc_real = torch.tensor(vxc_real, dtype=torch.float64)
    else:
        vxc_real = vxc_real.double()

    # Reshape v_XC_true if necessary
    if vxc_real.ndim == 1:
        vxc_real = vxc_real.to(device).view(-1, 1)

    Psi_tensor_tilde = orthogonalize_basis(Psi_tensor, S, wt)

    if verify:
        verify_orthonormality(Psi_tensor_tilde,wt,1e-5)

    # psi_functions: (M, N), v_XC_true: (M, 1), w: (M, N)
    integrand = (Psi_tensor_tilde.conj() * wt) * vxc_real   # Resulting shape: (M, N)
    C = torch.sum(integrand, dim=0)  # Shape: (N,)

    if write==True:
       torch.save(C, os.path.join(mol_dir,"C_tensor.pt"))
    return C


def create_coeff(device,nlm=['100','200','211','210','21-1'],dataset_file='dataset_small.csv', 
                 geom_folder='dataset'):
    import csv
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            if not os.path.exists(os.path.join(mol_dir,"C_tensor.pt")):
                get_coeff(mol_dir,device,nlm)
                print(f"Creating coefficient file in {mol_dir}")
            else:
                print(f"File already exists. Skipping save.")
    return

def delete_coeff(dataset_file='dataset_small.csv', 
                 geom_folder='dataset'):
    import csv
    datasetfile = os.path.join(geom_folder,dataset_file)
    with open(datasetfile, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            molecule_id = row[0]
            mol_dir = os.path.join(geom_folder,molecule_id)
            if os.path.exists(os.path.join(mol_dir,"C_tensor.pt")):
                os.remove(os.path.join(mol_dir,"C_tensor.pt"))
                print(f"Deleted coefficient file in {mol_dir}")
            else:
                print("Coefficient file does not exist. Skipping delete")
    return

def load_coeff(batch,device,geom_folder='dataset',verbose=0):
    num_graphs = batch.ptr.size(0) - 1
    C_list = []
    for graph_idx in range(num_graphs):
        molecule_name = batch.name[graph_idx]
        if verbose>0:
            print(f"Loading coefficients of vxc for {molecule_name}")
        coeff_file = os.path.join(geom_folder,molecule_name,"C_tensor.pt")
        C_mol = torch.load(coeff_file, map_location=device)
        C_list.append(C_mol)
    C_tensor = torch.cat(C_list,dim=0)
    return C_tensor

def load_coeff_v2(batch_name_tuple, device,geom_folder='dataset',verbose=0):
    num_graphs = len(batch_name_tuple)
    C_list = []
    for graph_idx in range(num_graphs):
        molecule_name = batch_name_tuple[graph_idx]
        if verbose>0:
            print(f"Loading coefficients of vxc for {molecule_name}")
        coeff_file = os.path.join(geom_folder,molecule_name,"C_tensor.pt")
        C_mol = torch.load(coeff_file, map_location=device)
        C_list.append(C_mol)
    C_tensor = torch.cat(C_list,dim=0)
    return C_tensor

def compare_plot_vxc(lambda_vxc_pred,batch,device,nlm=['100','200','211','210','21-1'],geom_folder="dataset", plot=True, save=False, save_dir=None):
    num_graphs = batch.ptr.size(0) - 1
    for graph_idx in range(num_graphs):
        molecule_name = batch.name[graph_idx]
        mol_dir = os.path.join(geom_folder,molecule_name)
        geom_file = os.path.join(mol_dir,"GEOM")
        vxc_file = os.path.join(mol_dir,"vxc")
        vxc_real = load_density(vxc_file)
        XYZ_orig = load_vxc_coordinates(vxc_file)
        atoms = read_geom_file(geom_file)
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
        Psi_list = []
        Nelec = 0
        for i,atom in enumerate(atoms):
            element, _ = atom
            Nelec += element_atomic_numbers[element]
            coords_at = coords[i]
            atom_wt = partial_weights[:,i]
            Psi_atomic = Psi_matrix(element,XYZ,coords_at,atom_wt,device,nlm)
            Psi_list.append(Psi_atomic)
        Psi_tensor = torch.cat(Psi_list,dim=1)
        # Psi_tensor = Psi_tensor.to(device)

        overlap_file = os.path.join(mol_dir,"S_tensor.pt")
        if os.path.exists(overlap_file):
            S = torch.load(overlap_file, map_location=device)
        else:
            S = get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device, nlm,write=True)

        Psi_tensor_tilde = orthogonalize_basis(Psi_tensor, S, wt)
        if Psi_tensor_tilde.dtype != lambda_vxc_pred.dtype:
            lambda_vxc_pred = lambda_vxc_pred.to(Psi_tensor_tilde.dtype)
        vxc_pred = torch.sum(Psi_tensor_tilde * lambda_vxc_pred, dim=1)#.to(device)

        del Psi_list, Psi_atomic, Psi_tensor
        del S, Psi_tensor_tilde
        del wt
        torch.cuda.empty_cache()

        if not torch.is_tensor(vxc_real):
            vxc_real = torch.tensor(vxc_real, dtype=torch.float64)
        else:
            vxc_real = vxc_real.double()

        rho_wf = load_density(os.path.join(geom_folder,molecule_name,"rho_wf"))
        wt = load_density(os.path.join(geom_folder,molecule_name,"gridwts"))

        if not torch.is_tensor(rho_wf):
            rho_wf = torch.tensor(rho_wf, dtype=torch.float64)
        else:
            rho_wf = rho_wf.double()
        if not torch.is_tensor(wt):
            wt = torch.tensor(wt, dtype=torch.float64)
        else:
            wt = wt.double()

        numerator = torch.sum((vxc_pred - vxc_real)**2 * rho_wf**2 * wt)
        denominator = torch.sum(vxc_real**2 * rho_wf**2 * wt)

        e1_metric = torch.sqrt(numerator) / torch.sqrt(denominator)

        print(f"Nelec: {Nelec}")
        print(f"e1 metric for vxc of molecule {molecule_name} is {e1_metric}")

        if plot:
            grid_coords = np.array(XYZ, dtype=float)

            # Filter coordinates where x and y are near zero
            mask = np.abs(grid_coords[:, 2]) < 1e-8
            mask &= np.abs(grid_coords[:, 1]) < 1e-8

            Z = grid_coords[mask, 0]

            vxc_tensor_real = vxc_real
            vxc_array_real = vxc_tensor_real.cpu().numpy()
            V_real = vxc_array_real[mask]

            vxc_tensor_pred = vxc_pred.detach()
            vxc_array_pred = vxc_tensor_pred.cpu().numpy()
            V_pred = vxc_array_pred[mask]

            sorted_indices = np.argsort(Z)
            Z = Z[sorted_indices]
            V_real = V_real[sorted_indices]
            V_pred = V_pred[sorted_indices]
            
            plt.plot(Z, V_real, color='red', linestyle='solid', markersize=2, marker='o', label="Real")
            plt.plot(Z, V_pred, color='blue', linestyle='solid', markersize=2, marker='o', label="Pred")
            plt.title(molecule_name)
            plt.legend()
            if save:
                plt.savefig(save_dir, dpi=500)
            plt.close()

    return e1_metric, Nelec

def compare_plot_vxc_v2(lambda_vxc_pred,batch_name_tuple,device,nlm=['100','200','211','210','21-1'],geom_folder="dataset",plot=True, save=False, save_dir=None):
    num_graphs = len(batch_name_tuple)
    for graph_idx in range(num_graphs):
        molecule_name = batch_name_tuple[graph_idx]
        print(molecule_name)
        mol_dir = os.path.join(geom_folder,molecule_name)
        geom_file = os.path.join(mol_dir,"GEOM")
        vxc_file = os.path.join(mol_dir,"vxc")
        vxc_real = load_density(vxc_file)
        XYZ_orig = load_vxc_coordinates(vxc_file)
        atoms = read_geom_file(geom_file)
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
        Psi_list = []
        Nelec = 0
        for i,atom in enumerate(atoms):
            element, _ = atom
            Nelec += element_atomic_numbers[element]
            coords_at = coords[i]
            atom_wt = partial_weights[:,i]
            Psi_atomic = Psi_matrix(element,XYZ,coords_at,atom_wt,device,nlm)
            Psi_list.append(Psi_atomic)
        Psi_tensor = torch.cat(Psi_list,dim=1)
        Psi_tensor = Psi_tensor.to(device)

        overlap_file = os.path.join(mol_dir,"S_tensor.pt")
        if os.path.exists(overlap_file):
            S = torch.load(overlap_file, map_location=device)
        else:
            S = get_overlap_mol(mol_dir, atoms, XYZ, coords, partial_weights, wt, device, nlm,write=True)

        Psi_tensor_tilde = orthogonalize_basis(Psi_tensor, S, wt)
        if Psi_tensor_tilde.dtype != lambda_vxc_pred.dtype:
            lambda_vxc_pred = lambda_vxc_pred.to(Psi_tensor_tilde.dtype) 
        vxc_pred = torch.sum(Psi_tensor_tilde * lambda_vxc_pred, dim=1)#.to(device)

        del Psi_list, Psi_atomic, Psi_tensor
        del S, Psi_tensor_tilde
        del wt
        torch.cuda.empty_cache()

        vxc_real = torch.from_numpy(vxc_real)#.to(device)

        rho_wf = load_density(os.path.join(geom_folder,molecule_name,"rho_wf"))
        wt = load_density(os.path.join(geom_folder,molecule_name,"gridwts"))

        if not torch.is_tensor(rho_wf):
            rho_wf = torch.tensor(rho_wf, dtype=torch.float64)
        else:
            rho_wf = rho_wf.double()
        if not torch.is_tensor(wt):
            wt = torch.tensor(wt, dtype=torch.float64)
        else:
            wt = wt.double()

        numerator = torch.sum((vxc_pred - vxc_real)**2 * rho_wf**2 * wt)
        denominator = torch.sum(vxc_real**2 * rho_wf**2 * wt)

        e1_metric = torch.sqrt(numerator) / torch.sqrt(denominator)

        print(f"Nelec: {Nelec}")
        print(f"e1 metric for vxc of molecule {molecule_name} is {e1_metric}")

        if plot:
            grid_coords = np.array(XYZ, dtype=float)

            # Filter coordinates where x and y are near zero
            mask = np.abs(grid_coords[:, 0]) < 1e-8
            mask &= np.abs(grid_coords[:, 1]) < 1e-8

            Z = grid_coords[mask, 2]
            # vxc_tensor_real = torch.tensor(vxc_real)
            vxc_tensor_real = vxc_real
            vxc_array_real = vxc_tensor_real.cpu().numpy()
            V_real = vxc_array_real[mask]

            # vxc_tensor_pred = torch.tensor(vxc_pred)
            vxc_tensor_pred = vxc_pred.detach()
            vxc_array_pred = vxc_tensor_pred.cpu().numpy()
            V_pred = vxc_array_pred[mask]

            sorted_indices = np.argsort(Z)
            Z = Z[sorted_indices]
            V_real = V_real[sorted_indices]
            V_pred = V_pred[sorted_indices]
            
            plt.plot(Z, V_real, color='red', linestyle='solid', markersize=2, marker='o', label="Real")
            plt.plot(Z, V_pred, color='blue', linestyle='solid', markersize=2, marker='o', label="Pred")
            plt.title(molecule_name)
            plt.legend()
            if save:
                plt.savefig(save_dir, dpi=500)
            plt.show()
    return e1_metric, Nelec

def calc_vxc_loss_coeff(lambda_pred, lambda_real, batch, device, nlm=['100','200','211','210','21-1'], geom_folder='dataset'):
    num_graphs = batch.ptr.size(0) - 1
    delta_lambda = lambda_pred - lambda_real
    loss = 0
    for graph_idx in range(num_graphs): # this will not work if the tensors are flattened
        start_idx = batch.ptr[graph_idx].item()
        end_idx = batch.ptr[graph_idx + 1].item()
        delta_lambda_graph = delta_lambda[start_idx:end_idx]
        molecule_name = batch.name[graph_idx]
        nlm_desc = read_descriptor_file(os.path.join(geom_folder,molecule_name,"c_nlm.csv"))
        atoms = read_geom_file(os.path.join(geom_folder,molecule_name,"GEOM"))
        C_nlm = []
        counts={}
        for element, _ in atoms:
            counts[element] = counts.get(element, 0) + 1
            key = f"{element}{counts[element]}"
            C_nlm.append(nlm_desc[key])
        C_nlm = torch.tensor(C_nlm,dtype=torch.float64).to(device)
        product = torch.abs(delta_lambda_graph * C_nlm)
        loss = loss + torch.sum(product)
    return loss / num_graphs

def calc_vxc_loss_coeff_v2(lambda_pred, lambda_real, batch_name_tuple, device, nlm=['100','200','211','210','21-1'], geom_folder='dataset'):
    num_graphs = len(batch_name_tuple)
    delta_lambda = lambda_pred - lambda_real
    lambda_index = 0 # indexing lambda by molecule
    loss = 0
    for graph_idx in range(num_graphs): # this will not work if the tensors are flattened
        molecule_name = batch_name_tuple[graph_idx]
        nlm_desc = read_descriptor_file(os.path.join(geom_folder,molecule_name,"c_nlm.csv"))
        atoms = read_geom_file(os.path.join(geom_folder,molecule_name,"GEOM"))
        C_nlm = []
        counts={}
        for element, _ in atoms:
            counts[element] = counts.get(element, 0) + 1
            key = f"{element}{counts[element]}"
            C_nlm.append(nlm_desc[key])
        C_nlm = torch.tensor(C_nlm,dtype=torch.float64).to(device)
        delta_lambda_graph = delta_lambda[lambda_index: lambda_index+len(C_nlm)]
        lambda_index = lambda_index + len(C_nlm)
        product = torch.abs(delta_lambda_graph * C_nlm)
        loss = loss + torch.sum(product)
    return loss / num_graphs