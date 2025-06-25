# Mapping from element symbols to atomic numbers
element_atomic_numbers = {
    'H': 1,  'He': 2,
    'Li': 3, 'Be': 4, 'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,  'Ne': 10,
    # Add more elements as needed
}

from .geom_utils import read_geom_file, read_descriptor_file, compute_distance, compute_distance_coord
from .create_graph import create_molecular_graph, MolecularDataset, create_dataset
from .evaluation import evaluation
from .read_config import read_config
from .grid_utils import load_vxc_coordinates, load_density, extract_nlm, make_basis, Psi_matrix, becke_weight, rotate_grid, nuclear_charge_moment, diagonalize_moment_tensor
from .descriptors import create_descriptor, delete_descriptor, create_descriptors_molecule, convert_descriptors_to_format, save_descriptors_to_csv, project_density_integrate
from .vxc import pred_vxc, real_vxc, create_coeff, delete_coeff, load_coeff, compare_plot_vxc, calc_vxc_loss_coeff, calc_vxc_molecule, load_coeff_v2, calc_vxc_loss_coeff_v2, get_coeff, calc_vxc_loss_coeff_old, compare_plot_vxc_v2
from .overlap import get_overlap_mol, orthogonalize_basis, verify_orthonormality, create_overlap, delete_overlap, compute_overlap_matrix, print_overlap
from .create_concat_dataset import MolecularDataset2
