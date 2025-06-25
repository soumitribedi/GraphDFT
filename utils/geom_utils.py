import csv
import torch

def read_geom_file(filename):
    """
    Read a geometry file and parse atomic positions.

    Each line of the file is expected to contain an element symbol followed by
    three Cartesian coordinates (x, y, z), separated by whitespace. For example:
        H  0.00000  0.00000  0.00000

    Args:
        filename (str): Path to the geometry file to read.

    Returns:
        List[Tuple[str, Tuple[float, float, float]]]:
            A list of atoms, where each atom is represented as a tuple:
            (element_symbol, (x, y, z)). Coordinates are Python floats.

    Raises:
        FileNotFoundError: If the specified file cannot be opened.
        ValueError: If any line does not have exactly four fields or if
            coordinate values cannot be converted to float.
    """
    atoms = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            element = parts[0]
            coords = torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])], dtype=torch.float64)
            atoms.append((element, tuple(coords.tolist())))
    return atoms

def read_descriptor_file(filename):
    """
    Read a CSV descriptor file and return per-atom descriptor tensors.

    The file is expected to have a header row followed by rows where:
      - Column 0 is the element symbol (e.g., "C", "H", "O").
      - Column 1 may be the atomic number.
      - Remaining columns are numeric descriptor values.

    Each element occurrence is numbered sequentially (e.g., "C1", "C2", "H1", …),
    and its corresponding descriptor vector is stored as a torch.float64 tensor.

    Args:
        filename (str): Path to the CSV file containing descriptors.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping keys of the form
        "<element><count>" (e.g., "C1", "O3") to a 1D torch.Tensor of dtype
        torch.float64 containing the descriptor values for that atom.

    Raises:
        FileNotFoundError: If the specified file cannot be opened.
        ValueError: If any descriptor value cannot be converted to float.
    """
    descriptors = {}
    counts = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            element = row[0]
            counts[element] = counts.get(element, 0) + 1
            key = f"{element}{counts[element]}"
            start_col = 2
            values = torch.tensor([float(x) for x in row[start_col:]], dtype=torch.float64)
            descriptors[key] = values
    return descriptors


def compute_distance(atom1, atom2):
    """
    Compute the Euclidean distance between two atoms.

    Each atom is represented as a tuple containing:
        - element symbol (str) at index 0
        - coordinates (tuple of three floats) at index 1

    Args:
        atom1 (Tuple[str, Tuple[float, float, float]]):
            First atom, e.g. ("H", (0.0, 0.0, 0.0)).
        atom2 (Tuple[str, Tuple[float, float, float]]):
            Second atom, e.g. ("O", (0.0, 0.0, 1.0)).

    Returns:
        torch.Tensor:
            A zero-dimensional tensor (scalar) of dtype torch.float64
            containing the Euclidean distance between atom1 and atom2.

    Raises:
        ValueError: If the coordinate tuples are not of length 3 or
            cannot be converted to floats.
    """
    coords1 = torch.tensor(atom1[1], dtype=torch.float64)
    coords2 = torch.tensor(atom2[1], dtype=torch.float64)
    distance = torch.norm(coords2 - coords1, p=2)
    return distance

def compute_distance_coord(coords1, coords2):
    """
    Compute the Euclidean (L2) distance between two coordinate vectors.

    This function accepts two coordinate inputs of the same shape and returns
    the scalar Euclidean distance between them:
        distance = ||coords2 - coords1||₂

    Args:
        coords1 (torch.Tensor or Sequence[float]): A 1D tensor or sequence
            of floats representing the first point’s coordinates (e.g., (x1, y1, z1)).
        coords2 (torch.Tensor or Sequence[float]): A 1D tensor or sequence
            of floats representing the second point’s coordinates (e.g., (x2, y2, z2)).
            Must be the same shape as `coords1`.

    Returns:
        torch.Tensor:
            A zero-dimensional tensor (scalar) of dtype torch.float64 containing
            the Euclidean distance between the two coordinate vectors.

    Raises:
        ValueError: If `coords1` and `coords2` have different shapes or cannot
            be broadcast to a common shape.
    """
    distance = torch.norm(coords2 - coords1, p=2)
    return distance
