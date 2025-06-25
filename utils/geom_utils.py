import csv
import torch

def read_geom_file(filename):
    """
    Parse a simple geometry file into a list of atomic symbols and coordinates.

    Each line of the file is expected to have the format:
        Element X Y Z
    where:
      - Element is a chemical symbol (e.g., 'H', 'C', 'O').
      - X, Y, Z are floating-point Cartesian coordinates.

    Parameters
    ----------
    filename : str
        Path to the geometry file.

    Returns
    -------
    List[Tuple[str, Tuple[float, float, float]]]
        A list where each entry is a tuple:
        (element_symbol, (x, y, z)),
        with coordinates returned as Python floats.

    Raises
    ------
    ValueError
        If any line does not have exactly four whitespace-separated fields
        or if coordinate conversion to float fails.
    IOError
        If the file cannot be opened.
    """
    atoms = []
    with open(filename, 'r') as file:
        for lineno, line in enumerate(file, start=1):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Line {lineno}: expected 4 fields, got {len(parts)}")
            element = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError as e:
                raise ValueError(f"Line {lineno}: could not parse coordinates") from e
            coords = torch.tensor([x, y, z], dtype=torch.float64)
            atoms.append((element, (x, y, z)))
    return atoms


def read_descriptor_file(filename):
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
