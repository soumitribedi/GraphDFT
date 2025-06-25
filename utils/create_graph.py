from .geom_utils import read_geom_file, read_descriptor_file, compute_distance
import torch
from torch_geometric.data import Data
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr
import os
import csv
from torch_geometric.data import InMemoryDataset
from .create_concat_dataset import MolecularDataset2

def create_molecular_graph(geom_file, descriptor_file, target, molecule_name):
    """
    Creates a molecular graph from geometry and descriptor files.

    This function constructs a molecular graph representation using the information from a geometry file and 
    a descriptor file. It generates node features, edge indices, and edge attributes based on atomic coordinates
    and distances between atoms. A PyTorch Geometric `Data` object is returned, which includes node features, 
    edge information, and target values.

    Parameters:
    ----------
    geom_file : str
        The path to the geometry file containing atomic element symbols and their Cartesian coordinates.
        Each line in the file should have the format: `Element x y z`.

    descriptor_file : str
        The path to the descriptor file containing atomic element symbols and their corresponding feature vectors.
        Each line should have the format: `Element feature1 feature2 ... featureN`.

    target : float
        The target value associated with the molecular graph.

    molecule_name : str
        The name of the molecule being represented.

    Returns:
    -------
    Data
        A PyTorch Geometric `Data` object containing:
        - `x`: A tensor of node features.
        - `edge_index`: A tensor of edge indices.
        - `edge_attr`: A tensor of edge attributes (distances).
        - `y`: A tensor containing the target value.
        - `coords`: A tensor of atom coordinates.
        - `name`: The name of the molecule.

    Raises:
    ------
    ValueError
        If a descriptor is not found for an atom's element in the descriptor file.

    Example:
    -------
    >>> data = create_molecular_graph('geom.csv', 'descriptors.csv', 0.987, 'Molecule_A')
    >>> print(data)
    Data(x=[10, 50], edge_index=[2, 18], edge_attr=[18, 1], y=[0.987], coords=[10, 3], name='Molecule_A')

    Notes:
    -----
    - The function assumes that the files are correctly formatted and that the descriptors match the elements
      in the geometry file.
    - If there are errors in file formats or missing descriptors, appropriate exceptions will be raised.
    """
    # Read input files
    atoms = read_geom_file(geom_file)
    descriptors = read_descriptor_file(descriptor_file)

    # Create node features
    node_features = []
    counts={}
    for element, _ in atoms:
        counts[element] = counts.get(element, 0) + 1
        key = f"{element}{counts[element]}"
        node_features.append(descriptors[key])
    
    if len(node_features) != len(atoms):
        missing_elements = set(element for element, _ in atoms if element not in descriptors)
        raise ValueError(f"No descriptor found for elements: {', '.join(missing_elements)}")

    # Extract coordinates
    coords = torch.tensor([atom[1] for atom in atoms], dtype=torch.float64)

    # Initialize edge lists
    num_atoms = len(atoms)
    edge_index = []
    edge_attr = []

    # Create edges based on distance
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = compute_distance(atoms[i], atoms[j])
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[distance], [distance]])

    # Convert to tensors
    x = torch.stack(node_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float64)
    y = torch.tensor([target], dtype=torch.float64)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, coords=coords, name=molecule_name)
    return data

class MolecularDataset(InMemoryDataset):
    """
    A custom PyTorch Geometric dataset for molecular graph data.

    This dataset reads molecular graph data from raw files, processes it into PyTorch Geometric 
    data objects, and stores the processed data for efficient access. It uses geometry files and 
    descriptor files to create molecular graphs and targets based on a dataset file that maps 
    molecule IDs to target values.

    Parameters:
    ----------
    root : str
        The root directory where the processed data will be saved.
        
    dataset_file : str
        The name of the file containing molecule IDs and target values, relative to `geom_folder`.
        
    geom_folder : str
        The directory containing geometry files for each molecule.
        
    descriptor_folder : str, optional
        The directory containing descriptor files. Default is `None`.
        
    descriptor_file : str, optional
        The file containing descriptor vectors for different elements, relative to `descriptor_folder`.
        If `None`, per-molecule descriptor files are expected. Default is `None`.
        
    transform : callable, optional
        A function/transform that takes a PyTorch Geometric `Data` object and returns a transformed version.
        Default is `None`.
        
    pre_transform : callable, optional
        A function/transform that takes a PyTorch Geometric `Data` object and returns a pre-transformed version.
        Default is `None`.

    Attributes:
    ----------
    geom_folder : str
        The directory containing geometry files.
        
    descriptor_folder : str or None
        The directory containing descriptor files.
        
    dataset_file : str
        The path to the dataset file.
        
    descriptor_file : str or None
        The path to the descriptor file, if using a global descriptor file.

    Methods:
    -------
    raw_file_names
        Returns the list of raw file names required by the dataset.
        
    processed_file_names
        Returns the list of processed file names.
        
    download
        No-op as no files are downloaded in this dataset implementation.
        
    process
        Reads raw data, processes it into PyTorch Geometric Data objects, applies transformations, 
        and saves the processed data.
    """
    def __init__(self, root, dataset_file, geom_folder, descriptor_folder=None, descriptor_file=None, transform=None, pre_transform=None):
        self.geom_folder = geom_folder
        self.descriptor_folder = descriptor_folder
        self.descriptor_file = descriptor_file
        self.dataset_file = os.path.join(self.geom_folder, dataset_file)
        if self.descriptor_folder and self.descriptor_file:
            self.descriptor_file = os.path.join(self.descriptor_folder, descriptor_file)
        super(MolecularDataset, self).__init__(root, transform, pre_transform)
        with safe_globals([DataEdgeAttr]):
            self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        for attr in ['x', 'edge_attr', 'y', 'coords']:
            val = getattr(self.data, attr, None)
            if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                setattr(self.data, attr, val.to(torch.float64))

    @property
    def raw_file_names(self):
        return [self.dataset_file]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No files are downloaded, as they are expected to be present
        pass

    def process(self):
        data_list = []
        with open(self.dataset_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                molecule_id, target = row[0], float(row[1])
                geom_file = os.path.join(self.geom_folder, molecule_id, 'GEOM')
                molecule_name = molecule_id

                if self.descriptor_file:
                    # Using a global descriptor file
                    descriptor_file_path = self.descriptor_file
                else:
                    # Using per-molecule descriptor file
                    descriptor_file_path = os.path.join(self.geom_folder, molecule_id, 'descriptors.csv')
                try:
                    data = create_molecular_graph(geom_file, descriptor_file_path, target, molecule_name)
                    data_list.append(data)
                except Exception as e:
                    print(f"Error processing molecule {molecule_id}: {e}")
                

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # Explicitly cast float tensors to float64 to preserve precision
        for attr in ['x', 'edge_attr', 'y', 'coords']:
            val = getattr(data, attr, None)
            if isinstance(val, torch.Tensor) and val.dtype == torch.float32:
                setattr(data, attr, val.to(torch.float64))
        torch.save((data, slices), self.processed_paths[0])


def create_dataset(root='./data', 
                   dataset_file='dataset_small.csv', 
                   geom_folder='dataset', 
                   descriptor_folder=None, 
                   descriptor_file = None,
                   type = "v1"):
    """
    Creates and returns a MolecularDataset object.

    This function initializes a MolecularDataset object using the provided file and 
    folder paths. It supports both global descriptor files and per-molecule descriptor files.

    Parameters:
    -----------
    root : str, optional
        Root directory where data is stored (default is './data').
    
    dataset_file : str, optional
        Filename for the dataset CSV file (default is 'dataset_small.csv').
    
    geom_folder : str, optional
        Name of the folder containing geometric data (default is 'dataset').
    
    descriptor_folder : str or None, optional
        Name of the folder containing descriptor data. If `None`, per-molecule descriptors are expected.
        Default is `None`.
    
    descriptor_file : str or None, optional
        Filename for the global descriptors CSV file. If `None`, per-molecule descriptors are expected.
        Default is `None`.

    type : str, optional
        If "v1", using MolecularDataset(InMemoryDataset) to create datsaet.
        if "v2", using MolecularDataset2(torch.util.data.Dataset) to create dataset.

    Returns:
    --------
    dataset : MolecularDataset
        An instance of the MolecularDataset class initialized with the provided paths.

    Raises:
    -------
    FileNotFoundError
        If any of the specified files or directories do not exist.
    """
    print(f"Working on {dataset_file}")
    # Validate that the geometry folder exists
    if not os.path.isdir(geom_folder):
        raise FileNotFoundError(f"The geometry folder '{geom_folder}' does not exist.")
    
    # Validate that the dataset file exists
    dataset_path = os.path.join(geom_folder, dataset_file)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"The dataset file '{dataset_path}' does not exist.")

    # If descriptor_folder and descriptor_file are provided, validate them
    if descriptor_folder is not None and descriptor_file is not None:
        # Validate that the descriptor folder exists
        descriptor_folder_path = os.path.join(descriptor_folder)
        if not os.path.isdir(descriptor_folder_path):
            raise FileNotFoundError(f"The descriptor folder '{descriptor_folder}' does not exist.")

        # Validate that the descriptor file exists
        descriptor_file_path = os.path.join(descriptor_folder_path, descriptor_file)
        if not os.path.isfile(descriptor_file_path):
            raise FileNotFoundError(f"The descriptor file '{descriptor_file_path}' does not exist.")
    else:
        # Using per-molecule descriptors
        # Optionally validate that at least one molecule's descriptor file exists
        with open(dataset_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            try:
                first_row = next(reader)
                molecule_id = first_row[0]
                descriptor_file_path = os.path.join(geom_folder, molecule_id, 'descriptors.csv')
                if not os.path.isfile(descriptor_file_path):
                    print(f"Warning: Descriptor file 'descriptors.csv' not found in molecule folder '{molecule_id}'.")
            except StopIteration:
                print("Warning: Dataset file is empty; no molecules to validate.")

    # Create the dataset
    if type == "v1":
        dataset = MolecularDataset(
            root=root, 
            dataset_file=dataset_file, 
            geom_folder=geom_folder, 
            descriptor_folder=descriptor_folder, 
            descriptor_file=descriptor_file
        )
    elif type == "v2":
        dataset = MolecularDataset2(
            dataset_file=dataset_file, 
            geom_folder=geom_folder, 
            descriptor_folder=descriptor_folder, 
            descriptor_file=descriptor_file
        )
    else:
        raise ValueError("Dataset type no found.")
    return dataset