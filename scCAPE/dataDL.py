# -*- coding:utf-8 -*-
"""
name:DL Data
function:process the data into dataloader
"""


from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import scanpy as sc
import scipy

get_indx = lambda a, i: a[i] if a is not None else None


class Expdata:
    def __init__(
            self,
            data,
            perturbation_key=None,
            split_key="split"
    ):
        super(Expdata, self).__init__()

        data = sc.read_h5ad(data)
        self.perturbation_key = perturbation_key
        self.cell_names = data.obs_names

        if scipy.sparse.issparse(data.X):
            self.genes = torch.Tensor(data.X.A)
        else:
            self.genes = torch.Tensor(data.X)

        self.var_names = data.var_names

        if split_key in data.obs:
            pass
        else:
            print("Performing automatic train-test split with 0.2 ratio...")
            from sklearn.model_selection import train_test_split

            data.obs[split_key] = "train"
            # idx = list(range(len(data)))
            idx_train, idx_test = train_test_split(
                data.obs_names, test_size=0.20, random_state=42
            )
            data.obs.loc[idx_train, split_key] = "train"
            data.obs.loc[idx_test, split_key] = "test"

        if "control" in data.obs:
            pass
        else:
            ctrl_dummy = np.zeros_like(data.obs[perturbation_key])
            for i in range(len(ctrl_dummy)):
                if data.obs[perturbation_key][i] == 'control':
                    ctrl_dummy[i] = 1
            data.obs['control'] = ctrl_dummy.tolist()
        self.ctrl = data.obs["control"].values
        print(f"Assigned {sum(self.ctrl)} control cells.")

        if "condition_name" in data.obs:
            pass
        else:
            data.obs['condition_name'] = data.obs[perturbation_key]
        self.perts_fullname = data.obs["condition_name"]

        if 'rank_genes_groups_cov_all' in data.uns.keys():
            pass
        else:
            print('Performing DEG analysis...')
            DEG_dict = {}
            for type_name in data.obs['condition_name'].unique():
                if type_name != 'control':
                    subdat = data[data.obs[perturbation_key].isin([type_name, 'control']), :]
                    sc.tl.rank_genes_groups(subdat, groupby="condition", method="wilcoxon", use_raw=False)
                    DEG_dict[type_name] = subdat.uns['rank_genes_groups']['names'][type_name]
            data.uns['rank_genes_groups_cov_all'] = DEG_dict
        self.de_genes = data.uns["rank_genes_groups_cov_all"]

        if "cell_type" in data.obs:
            pass
        else:
            print('Performing leiden clustering...')
            sc.tl.leiden(data, resolution=0.6)
            data.obs.rename(columns={'leiden': 'cell_type'}, inplace=True)
        self.cell_state = data.obs["cell_type"]
        self.cell_state_code = self.cell_state.astype("category").cat.codes.values.astype(int)
        self.cell_state_code = torch.from_numpy(self.cell_state_code).long()

        self.perts_names = np.array(data.obs[perturbation_key].values)
        self.perts_names_unique = np.array(data.obs[perturbation_key].values.unique())
        self.perts_nums = len(self.perts_names_unique)

        encoder_pert = OneHotEncoder(sparse=False)
        encoder_pert.fit(self.perts_names_unique.reshape(-1, 1))

        self.name2ohe = dict(
            zip(
                self.perts_names_unique,
                encoder_pert.transform(self.perts_names_unique.reshape(-1, 1)),
            )
        )
        # print(self.name2ohe)

        lb = []
        for i in self.name2ohe.keys():
            lb.append(int(np.where(self.name2ohe[i] == 1)[0]))
        self.name2label = dict(zip(self.name2ohe.keys(), lb))

        # print(self.name2label)

        perts = []
        for i, comb in enumerate(self.perts_names):
            pertohe = self.name2ohe[comb]
            perts.append(pertohe)

        self.perts = torch.Tensor(np.array(perts))

        self.num_genes = self.genes.shape[1]
        self.num_perts = len(self.perts_names_unique) if self.perts is not None else 0
        self.is_control = data.obs["control"].values.astype(bool)
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs["control"] == 1)[0].tolist(),
            "treated": np.where(data.obs["control"] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist()
        }

    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, i):
        return (
            self.genes[i],
            get_indx(self.perts, i),
            get_indx(self.cell_state_code, i))


class SubDataset:
    """
    Subsets a `ExpData` by selecting the samples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.perturbation_key = dataset.perturbation_key

        self.name2ohe = dataset.name2ohe
        self.name2label = dataset.name2label
        self.perts_names_unique = dataset.perts_names_unique
        self.perts_fullname = get_indx(dataset.perts_fullname, indices)
        self.genes = dataset.genes[indices]
        self.perts = get_indx(dataset.perts, indices)
        self.cell_names = get_indx(dataset.cell_names, indices)
        self.cell_state = get_indx(dataset.cell_state, indices)
        self.cell_state_code = get_indx(dataset.cell_state_code, indices)
        self.perts_names = get_indx(dataset.perts_names, indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        # self.ctrl_name = 'control'
        self.num_genes = dataset.num_genes
        self.num_perts = dataset.num_perts
        self.is_control = dataset.is_control[indices]

    def __getitem__(self, i):
        return (
            self.genes[i],
            get_indx(self.perts, i),
            get_indx(self.cell_state_code, i)
        )

    def subset_condition(self, control=True):
        idx = np.where(self.is_control == control)[0].tolist()
        return SubDataset(self, idx)

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
        data: str,
        perturbation_key: Union[str, None],
        split_key: Union[str, None],
        return_dataset: bool = False,
):
    dataset = Expdata(
        data, perturbation_key, split_key
    )

    splits = {
        "training": dataset.subset("train", "all"),
        "test": dataset.subset("test", "all"),
        "all": dataset.subset("all", "all")
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits


def main():
    # db = load_dataset_splits(data='4ps_leiden.h5ad', perturbation_key='condition',split_key=None)
    db = load_dataset_splits(data='data/pbmc_processed.h5ad', perturbation_key='condition', split_key=None)
    db.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                db["training"],
                batch_size=10,
                shuffle=True,
            )
        }
    )
    for data in db["loader_tr"]:
        genes, perts, cell_states = data[0], data[1], data[2]
        break

    print('sample:', genes.shape, perts.shape, perts, cell_states)
    print(db["training"].num_genes, db["training"].num_perts)
    print(db["training"].name2label)


if __name__ == '__main__':
    main()
