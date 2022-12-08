import csv
import functools
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from torch.utils.data import Dataset
from torch_geometric.data import Data


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomFeatureEncoder(object):
    def __init__(self):
        self.__atoms = []
        for atom_number in range(1, 101):
            self.__atoms.append(Element.from_Z(atom_number))
        self.__continuous_list = [[], [], [], [], []]
        self.__get_continuous_fea()
        self.__disperse_list = [[], [], [], [], []]
        self.__block_list = ['s', 'p', 'd', 'f']
        self.__disperse()

    def __get_continuous_fea(self):
        continuous_fea = ['X', 'van_der_waals_radius', 'ionization_energy', 'electron_affinity', 'molar_volume']
        for atom in self.__atoms:
            self.__continuous_list[0].append(atom.X if atom.number not in [2, 10, 18] else 0)
            for i in range(1, 5):
                fea = getattr(atom, continuous_fea[i])
                self.__continuous_list[i].append(fea if fea is not None else 80)

    def __disperse(self):
        for i in range(len(self.__continuous_list)):
            self.__disperse_list[i] = pd.cut(self.__continuous_list[i],
                                             10, right=True, labels=False,
                                             retbins=False, precision=3,
                                             include_lowest=False, duplicates='raise')

    def get_atom_features(self, atom_number):
        atom = Element.from_Z(atom_number)
        f1 = np.array([atom_number - 1,
                       atom.group - 1,
                       atom.row - 1,
                       self.__block_list.index(atom.block)])
        f2 = np.array([self.__disperse_list[i][atom.number - 1] for i in range(5)])
        atom_feature = np.concatenate((f1, f2))
        return atom_feature

    # def get_atom_features(self, atom_number):
    #     atom = Element.from_Z(atom_number)
    #     f1 = np.array([atom.group - 1,
    #                    atom.row - 1,
    #                    self.__block_list.index(atom.block)])
    #     f2 = np.array([self.__disperse_list[i][atom.number - 1] for i in range(5)])
    #     atom_feature = np.concatenate((f1, f2))
    #     return atom_feature


    @staticmethod
    def get_full_dims():
        return np.array([100, 18, 9, 4, 10, 10, 10, 10, 10])


class CIFData(Dataset):

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        self.afe = AtomFeatureEncoder()
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def format_adj_matrix(self, adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))

        atom_fea = np.vstack([[self.afe.get_atom_features(crystal[i].specie.number) for i in range(len(crystal))]])
        atom_fea = torch.LongTensor(atom_fea)
        ### for use only atomic number
        #atom_fea = np.hstack([[crystal[i].specie.number for i in range(len(crystal))]])
        #atom_fea = torch.LongTensor(atom_fea).unsqueeze(1)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea = nbr_fea.view(-1, 41)
        nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        target = torch.Tensor([float(target)])
        num_atoms = atom_fea.shape[0]
        orig_atom_fea_len = atom_fea.shape[1]
        crystal_graph = Data(x=atom_fea, edge_index=nbr_fea_idx, edge_attr=nbr_fea, y=target,
                             cif_id=cif_id, num_atoms=num_atoms, orig_atom_fea_len=orig_atom_fea_len)
        return crystal_graph
