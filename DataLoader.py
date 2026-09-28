import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.metrics.all_metrics import MSE, PearsonCorrelation, MAE
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import subsample_idxs, shuffle_arr
from collections import Counter

DATASET = '2009-17'
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']


DHS_SITES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
            41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
            51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0]


_SPLIT_DATA_60_TOTAL = {
    'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 36.0, 39.0, 40.0],
    'val':[54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 61.0, 62.0, 63.0, 64.0],
    'test':[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
            51.0, 52.0, 53.0],
    'val_id':[37.0, 38.0, 60.0, 65.0, 66.0, 67.0, 68.0, 69.0],
    'test_id':[35.0]
}

_SPLIT_DATA_40_TOTAL = {
    'train':[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0,],
    'val':[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
           25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,],
    'test':[37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 65.0],
    'val_id':[34.0, 35.0, 36.0, 64.0],
    'test_id':[46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
               56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 66.0, 67.0, 
               68.0]
}

_SPLIT_DATA_60 = {
    'train': _SPLIT_DATA_60_TOTAL['train']+_SPLIT_DATA_60_TOTAL['val_id']+_SPLIT_DATA_60_TOTAL['test_id'],
    'val':_SPLIT_DATA_60_TOTAL['val'],
    'test':_SPLIT_DATA_60_TOTAL['test'],
}

_SPLIT_DATA_40 = {
    'train': _SPLIT_DATA_40_TOTAL['train'],# + _SPLIT_DATA_40_TOTAL['val_id'] +_SPLIT_DATA_40_TOTAL['test_id'],
    'val':_SPLIT_DATA_40_TOTAL['val'],# +_SPLIT_DATA_40_TOTAL['val_id'],
    'test':_SPLIT_DATA_40_TOTAL['test'], # +_SPLIT_DATA_40_TOTAL['test_id'],
}

_SPLIT_DATA_PROPOSED_BALANCED = {
    "train": [
        1.0, 3.0, 4.0, 7.0, 8.0, 12.0, 13.0, 17.0, 21.0, 24.0,
        27.0, 28.0, 29.0, 30.0, 32.0, 37.0, 41.0, 43.0, 45.0,
        46.0, 47.0, 50.0, 51.0, 52.0, 58.0, 60.0, 61.0, 63.0, 64.0,
    ],

    "val": [
        6.0, 10.0, 14.0, 15.0, 18.0, 19.0, 22.0, 31.0, 33.0,
        39.0, 40.0, 44.0, 49.0, 54.0, 62.0, 65.0, 69.0,
    ],

    "test": [
        2.0, 5.0, 9.0, 11.0, 16.0, 20.0, 23.0, 26.0, 34.0,
        36.0, 42.0, 48.0, 55.0, 56.0, 59.0,
    ],
    # Excluded from main train/val/test because it is a large older outlier site.
    "id_test": [
        35.0,
    ],
}

_SPLIT_DATA = _SPLIT_DATA_PROPOSED_BALANCED


# _SPLIT_DATA = {
#     'train': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
#             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
#             21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
#             31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
#             41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,],
#     #,"51.0", "52.0", "53.0", "54.0", "55.0"],
#     'val': [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
#             61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,],
#     'test': [69.0, 70.0] #64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0
# }

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands
# all images have already been mean subtracted and normalized (x - mean) / std

_MEANS_2009_17 = {
    'BLUE':  0.059183,
    'GREEN': 0.088619,
    'RED':   0.104145,
    'SWIR1': 0.246874,
    'SWIR2': 0.168728,
    'TEMP1': 299.078023,
    'NIR':   0.253074,
    'DMSP':  4.005496,
    'VIIRS': 1.096089,
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}

_STD_DEVS_2009_17 = {
    'BLUE':  0.022926,
    'GREEN': 0.031880,
    'RED':   0.051458,
    'SWIR1': 0.088857,
    'SWIR2': 0.083240,
    'TEMP1': 4.300303,
    'NIR':   0.058973,
    'DMSP':  23.038301,
    'VIIRS': 4.786354,
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}


# def split_by_countries(idxs, ood_countries, metadata):
#     countries = np.asarray(metadata['country'].iloc[idxs])
#     is_ood = np.any([(countries == country) for country in ood_countries], axis=0)
#     return idxs[~is_ood], idxs[is_ood]

def split_by_site(idxs, ood_sites, metadata):
    # print("idxs: ", idxs)
    # print("ood sites: ",ood_sites)
    sites = np.asarray(metadata['site'].iloc[idxs])
    # print("sites: ", sites)
    is_ood = np.any([(sites == site) for site in ood_sites], axis=0)
    return idxs[~is_ood], idxs[is_ood]

class OpenBHBDataset(WILDSDataset):
    """
    The PovertyMap poverty measure prediction dataset.
    This is a processed version of LandSat 5/7/8 satellite imagery originally from Google Earth Engine under the names `LANDSAT/LC08/C01/T1_SR`,`LANDSAT/LE07/C01/T1_SR`,`LANDSAT/LT05/C01/T1_SR`,
    nighttime light imagery from the DMSP and VIIRS satellites (Google Earth Engine names `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4` and `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`)
    and processed DHS survey metadata obtained from https://github.com/sustainlab-group/africa_poverty and originally from `https://dhsprogram.com/data/available-datasets.cfm`.

    Supported `split_scheme`:
        - 'official' and `countries`, which are equivalent
        - 'mixed-to-test'

    Input (x):
        224 x 224 x 8 satellite image, with 7 channels from LandSat and 1 nighttime light channel from DMSP/VIIRS. Already mean/std normalized.

    Output (y):
        y is a real-valued asset wealth index. Higher index corresponds to more asset wealth.

    Metadata:
        each image is annotated with location coordinates (noised for anonymity), survey year, urban/rural classification, country, nighttime light mean, nighttime light median.

    Website: https://github.com/sustainlab-group/africa_poverty

    Original publication:
    @article{yeh2020using,
        author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
        day = {22},
        doi = {10.1038/s41467-020-16185-w},
        issn = {2041-1723},
        journal = {Nature Communications},
        month = {5},
        number = {1},
        title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
        url = {https://www.nature.com/articles/s41467-020-16185-w},
        volume = {11},
        year = {2020}
    }

    License:
        LandSat/DMSP/VIIRS data is U.S. Public Domain.

    """
    _dataset_name = 'openBHB'
    _versions_dict = {
        '1.1': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xfc0aa86ad9af4eb08c42dfc40eacf094/contents/blob/',
            'compressed_size': 13_091_823_616}}

    def __init__(self, version=None, root_dir='/rhome/ssafa013/bigdata/data', download=False,  # 'data'
                 split_scheme='official',
                 use_ood_val=True):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        # self._original_resolution = (224, 224)
        print(self._data_dir)
        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}

        if split_scheme == 'official':
            split_scheme = 'sites'

        if split_scheme == 'mixed-to-test':
            self.oracle_training_set = True
        elif split_scheme in ['official', 'sites']:
            self.oracle_training_set = False
        else:
            raise ValueError("Split scheme not recognized")
        self._split_scheme = split_scheme

        fold = 'A'
        self.root = Path(self._data_dir)
        self.metadata = pd.read_csv(self.root / 'images/metadata.tsv', sep='\t')
        # country folds, split off OOD
        # country_folds = SURVEY_NAMES[f'2009-17{fold}']
        # print('site max:', self.metadata['site'].max(), 'min: ', self.metadata['site'].min())
        # print('study max:', self.metadata['study'].max(), 'min: ', self.metadata['study'].min())

        self.metadata["site"] = self.metadata["site"] - 1
        self.metadata["study"] = self.metadata["study"] - 1

        # Keep the actual 0-based site ID for splitting and later analysis.
        # The proposed split lists are based on this 0-based site ID.
        self.metadata["original_site"] = self.metadata["site"].astype(float)

        site_folds = _SPLIT_DATA
        self._split_array = -1 * np.ones(len(self.metadata), dtype=np.int64)

        all_idxs = np.arange(len(self.metadata))

        # Assign subjects by site. No site is shared between train/val/test.
        _, idxs_train = split_by_site(all_idxs, site_folds["train"], self.metadata)
        _, idxs_val   = split_by_site(all_idxs, site_folds["val"],   self.metadata)
        _, idxs_test  = split_by_site(all_idxs, site_folds["test"],  self.metadata)

        self._split_array[idxs_train] = self._split_dict["train"]
        self._split_array[idxs_val]   = self._split_dict["val"]
        self._split_array[idxs_test]  = self._split_dict["test"]

        # Optional excluded / ignored sites.
        # We assign site 35 to id_test so it is not unassigned,
        # but the training script never uses dataset.get_subset("id_test").
        if "id_test" in site_folds:
            _, idxs_id_test = split_by_site(all_idxs, site_folds["id_test"], self.metadata)
            self._split_array[idxs_id_test] = self._split_dict["id_test"]
        else:
            idxs_id_test = np.array([], dtype=int)

        if "id_val" in site_folds:
            _, idxs_id_val = split_by_site(all_idxs, site_folds["id_val"], self.metadata)
            self._split_array[idxs_id_val] = self._split_dict["id_val"]
        else:
            idxs_id_val = np.array([], dtype=int)
            
        # Safety check: every subject should belong to exactly one of train/val/test.
        if np.any(self._split_array < 0):
            missing_sites = sorted(
                self.metadata.loc[self._split_array < 0, "original_site"].unique().tolist()
            )

            missing_counts = (
                self.metadata.loc[self._split_array < 0]
                .groupby("original_site")
                .agg(
                    n=("participant_id", "count"),
                    age_mean=("age", "mean"),
                    age_std=("age", "std"),
                    age_min=("age", "min"),
                    age_max=("age", "max"),
                )
                .reset_index()
            )

            print("\nERROR: Some subjects were not assigned to any split.")
            print("Unassigned sites:", missing_sites)
            print("\nUnassigned site summary:")
            print(missing_counts.to_string(index=False))

            raise RuntimeError(
                "Split is incomplete. Every site must be assigned to one of: "
                "train, val, test, id_val, or id_test. "
                "If a site should be excluded from training/evaluation, put it in id_test."
            )
            
        print("New split counts:")
        unique, counts = np.unique(self._split_array, return_counts=True)
        print(dict(zip(unique, counts)))

        print("Train sites:", sorted(self.metadata.loc[idxs_train, "original_site"].unique().tolist()))
        print("Val sites:", sorted(self.metadata.loc[idxs_val, "original_site"].unique().tolist()))
        print("Test sites:", sorted(self.metadata.loc[idxs_test, "original_site"].unique().tolist()))
        print("ID/test excluded sites:", sorted(self.metadata.loc[idxs_id_test, "original_site"].unique().tolist())) 
        train_site_set = set(self.metadata.loc[idxs_train, "original_site"].unique().tolist())
        val_site_set = set(self.metadata.loc[idxs_val, "original_site"].unique().tolist())
        test_site_set = set(self.metadata.loc[idxs_test, "original_site"].unique().tolist())

        if train_site_set & val_site_set or train_site_set & test_site_set or val_site_set & test_site_set:
            raise RuntimeError(
                "Site overlap detected between train/val/test. "
                f"train∩val={train_site_set & val_site_set}, "
                f"train∩test={train_site_set & test_site_set}, "
                f"val∩test={val_site_set & test_site_set}"
            )
        # ------------------------------------------------------------
        # Domain labels for discriminator training
        # ------------------------------------------------------------
        # The discriminator is trained only on train sites.
        # Raw train site IDs are not contiguous, so remap them to:
        #   0, 1, 2, ..., num_train_domains - 1
        #
        # Val/test are OOD sites and should not be used for discriminator training.
        # We assign them domain_site = -1.
        # ------------------------------------------------------------

        train_sites = sorted(list(site_folds["train"]))
        self.train_sites = train_sites
        self.num_train_domains = len(train_sites)

        site_to_domain = {site: i for i, site in enumerate(train_sites)}

        self.metadata["domain_site"] = self.metadata["original_site"].map(site_to_domain)
        self.metadata["domain_site"] = self.metadata["domain_site"].fillna(-1).astype(int)

        print("Train site -> domain label mapping:")
        print(site_to_domain)
        print("num_train_domains:", self.num_train_domains)


        unique, counts = np.unique(self._split_array, return_counts=True)
        # print(dict(zip(unique, counts)))

        self._y_array = torch.from_numpy(np.asarray(self.metadata['age'])[:, np.newaxis]).float()
        self._y_size = 1
        # add site group field
        # Metadata column 0 is the discriminator/domain label.
        # For train subjects this is 0..num_train_domains-1.
        # For val/test OOD subjects this is -1.
        #
        # We also keep original_site as the last column for analysis/debugging.

        self._metadata_map = {
            "site": list(range(self.num_train_domains)),
            "original_site": sorted(self.metadata["original_site"].unique().tolist()),
        }

        self._metadata_array = torch.from_numpy(
            self.metadata[
                ["domain_site", "age", "study", "participant_id", "original_site"]
            ].astype(float).to_numpy()
        )

        self._metadata_fields = [
            "site",            # actually domain_site; kept as "site" so old training code still works
            "y",
            "study",
            "participant_id",
            "original_site",
        ]
        
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['study'])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        #print(idx)
        participant_id = self.metadata['participant_id'][idx]
        # print("participant_id", participant_id)
        # _preproc-quasiraw_T1w
        # _preproc-cat12vbm_desc-gm_T1w
        # img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
        try:
            img = np.load(self.root / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
        except FileNotFoundError:
            print("File not found!, using different root directory  ", participant_id)
            new_dir = os.path.join('/data/ssafa013/wildsTest/data', f'{self.dataset_name}_v{self.version}')
            img = np.load(Path(new_dir) / 'images' / f'sub-{participant_id}_preproc-cat12vbm_desc-gm_T1w.npy')
            pass
        img = torch.from_numpy(img).float()

        return img

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model
            - y_true (LongTensor): Ground-truth values
            - metadata (Tensor): Metadata
            - prediction_fn (function): Only None supported
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        assert prediction_fn is None, "PovertyMapDataset.eval() does not support prediction_fn"

        metrics = [MSE(), MAE()]

        all_results = {}
        all_results_str = ''
        for metric in metrics:
            results, results_str = self.standard_group_eval(
                metric,
                self._eval_grouper,
                y_pred, y_true, metadata)
            all_results.update(results)
            all_results_str += results_str
        return all_results, all_results_str
