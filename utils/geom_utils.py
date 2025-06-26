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
    """
    Parse a CSV file of atomic descriptors into a dictionary mapping keys to tensors.

    The CSV is expected to have a header row, which will be skipped. Each subsequent row
    should contain at least three columns:
        Element, <unused>, value1, value2, ..., valueN

    A running count is kept for each Element to produce keys of the form "<Element><count>",
    e.g. "H1", "C2", etc.

    Parameters
    ----------
    filename : str
        Path to the CSV descriptor file.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dict mapping each descriptor key ("Element"+count) to a 1D torch.Tensor
        (dtype=torch.float64) of the parsed descriptor values.

    Raises
    ------
    ValueError
        If any data row has fewer than 3 columns or if a descriptor value cannot be
        converted to float.
    IOError
        If the file cannot be opened.
    """
    descriptors = {}
    counts = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        try:
            next(reader)
        except StopIteration:
            return {}
        for lineno, row in enumerate(reader, start=2):
            if len(row) < 3:
                raise ValueError(f"Line {lineno}: expected at least 3 columns, got {len(row)}")
            element = row[0]
            counts[element] = counts.get(element, 0) + 1
            key = f"{element}{counts[element]}"
            try:
                values = torch.tensor([float(x) for x in row[2:]], dtype=torch.float64)
            except ValueError as e:
                raise ValueError(f"Line {lineno}: could not parse descriptor values") from e
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
        TypeError
            If any coordinate cannot be converted to float.
    """
    coords1 = torch.tensor(atom1[1], dtype=torch.float64)
    coords2 = torch.tensor(atom2[1], dtype=torch.float64)
    distance = torch.norm(coords2 - coords1, p=2)
    return distance

def compute_distance_coord(coords1, coords2):
    """
    Compute the Euclidean (L2) distance between two coordinate vectors.

    Args:
        coords1 (torch.Tensor or Sequence[float]): A 1D tensor
            of floats representing the first point’s coordinates (e.g., (x1, y1, z1)).
        coords2 (torch.Tensor or Sequence[float]): A 1D tensor
            of floats representing the second point’s coordinates (e.g., (x2, y2, z2)).
            Must be the same shape as `coords1`.

    Returns:
        torch.Tensor:
            A zero-dimensional tensor (scalar) of dtype torch.float64 containing
            the Euclidean distance ‖coords2 − coords1‖₂. The output dtype is the same as the input tensors’.

    Raises:
        TypeError
            If you pass in non-tensor inputs (e.g. lists/tuples) or if tensor
            elements cannot be converted to numeric types.
        RuntimeError
            If `coords1` and `coords2` have different shapes and cannot be
            broadcast together.
    """
    if not isinstance(coords1, torch.Tensor) or not isinstance(coords2, torch.Tensor):
        raise TypeError("Both coords1 and coords2 must be torch.Tensor")
    distance = torch.norm(coords2 - coords1, p=2)
    return distance
