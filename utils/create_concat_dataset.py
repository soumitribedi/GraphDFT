import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .geom_utils import read_descriptor_file, read_geom_file

ELEMENT_ORDER = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
ELEMENT_TO_IDX = {elem: i for i, elem in enumerate(ELEMENT_ORDER)}

class MolecularDataset2(Dataset):
    def __init__(self, dataset_file, geom_folder, max_atoms=6, desc_dim=9, descriptor_folder=None, descriptor_file=None, transform=None, target_transform=None):
        self.geom_folder = geom_folder
        self.descriptor_folder = descriptor_folder
        self.descriptor_file = descriptor_file
        self.dataset_file = os.path.join(self.geom_folder, dataset_file)
        if self.descriptor_folder and self.descriptor_file:
            self.descriptor_file = os.path.join(self.descriptor_folder, descriptor_file)
        self.transform = transform
        self.target_transform = target_transform
        self.max_atoms = max_atoms
        self.desc_dim = desc_dim
        self.df = pd.read_csv(self.dataset_file)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        molecule_id = self.df.iloc[idx, 0]
        # print(molecule_id)
        target = self.df.iloc[idx, 1]
        atoms = read_geom_file(os.path.join(self.geom_folder, molecule_id, 'GEOM'))

        if self.descriptor_file:
            # Using a global descriptor file
            descriptor_file_path = self.descriptor_file
        else:
            # Using per-molecule descriptor file
            descriptor_file_path = os.path.join(self.geom_folder, molecule_id, 'descriptors.csv')
        try:
            descriptors = read_descriptor_file(descriptor_file_path)
            atoms_sorted = sorted(
                atoms, 
                key=lambda x: ELEMENT_TO_IDX.get(x[0], float('inf'))
            )
            feature_tensor = torch.zeros(self.max_atoms, self.desc_dim, dtype=torch.float64)
            counts={}
            for i, (element, _) in enumerate(atoms_sorted):
                if i >= self.max_atoms:
                    break
                counts[element] = counts.get(element, 0) + 1
                key = f"{element}{counts[element]}"
                if key in descriptors:
                    feature_tensor[i] = torch.tensor(descriptors[key], dtype=torch.float64)
            concatenated_feature = feature_tensor.flatten()
        except Exception as e:
            print(f"Error processing molecule {molecule_id}: {e}")

        target = torch.tensor(target, dtype=torch.float64)
        if self.transform:
            concatenated_feature = self.transform(concatenated_feature)
        if self.target_transform:
            target = self.target_transform(target)
        return concatenated_feature, target, molecule_id
