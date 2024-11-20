import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import pickle


class STPPDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_prop, data_name):
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.data_prop = data_prop

    def setup(self,stage=None):
        if self.data_name == 'synthetic':
            file_name = 'data/synthetic_data/simulated_stpp_f1_u_deep_kernel_no_z.csv'
            data = pd.read_csv(file_name)
            events = torch.stack([torch.tensor(data.x, dtype=torch.float32), torch.tensor(data.y, dtype=torch.float32), torch.tensor(data.t, dtype=torch.float32), torch.tensor(data.z, dtype=torch.float32)], dim=1)
            self.data = torch.split(events, torch.unique(torch.tensor(data.sim_no), return_counts=True)[1].tolist(), dim=0)

        elif self.data_name == 'vancouver':
            file_name = 'data/real_data/vancouver_crime.csv.gz'
            data = pd.read_csv(file_name)
            events = torch.stack([torch.tensor(data.x, dtype=torch.float32), torch.tensor(data.y, dtype=torch.float32), torch.tensor(data.t, dtype=torch.float32), torch.tensor(data.z, dtype=torch.float32)], dim=1)
            # self.data = events
            self.data = torch.split(events, torch.unique(torch.tensor(data.seq), return_counts=True)[1].tolist(), dim=0)

        elif self.data_name == "collision":
            file_name = 'data/real_data/nypd_collision.pkl'
            data = pickle.load(open(file_name,"rb"))
            events_space = torch.stack([torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.float32), torch.tensor(data['t'], dtype=torch.float32)], dim=1)
            events = torch.cat([events_space,torch.tensor(data['z'], dtype=torch.float32)],dim=1)
            # self.data = events
            self.data = torch.split(events, torch.unique(torch.tensor(data['seq']), return_counts=True)[1].tolist(), dim=0)


        elif self.data_name == "compliants":
            file_name = 'data/real_data/nypd_compliants.pkl'
            data = pickle.load(open(file_name,"rb"))
            events_space = torch.stack([torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.float32), torch.tensor(data['t'], dtype=torch.float32)], dim=1)
            events = torch.cat([events_space,torch.tensor(data['z'], dtype=torch.float32)],dim=1)
            # self.data = events
            self.data = torch.split(events, torch.unique(torch.tensor(data['seq']), return_counts=True)[1].tolist(), dim=0)


        self.data = self.data[:int(len(self.data) * self.data_prop)]
        # self.train_data = self.data[:int(len(self.data) * 0.8)]
        # self.val_data = self.data[int(len(self.data) * 0.8):int(len(self.data)*0.9)]
        # self.test_data = self.data[int(len(self.data)*0.9):]

    def collate_fn_generator(self, mode="train"):
        def collate_fn(batch):
            batch_size = len(batch)
            if self.data_name == 'synthetic':
                out_batch = {'batch_size': batch_size,
                             'data': torch.cat(batch, dim=0),
                             'seq_lens': [s.shape[0] for s in batch],
                             'integration_domain': [[0, 100], [0, 100], [0, 100]]}
            else:

                if mode == "train":
                    filtered_batch = [b[b[:, 2] <= 50] for b in batch]
                    out_batch = {'batch_size': batch_size,
                                 'data': torch.cat(filtered_batch, dim=0),
                                 'seq_lens': [s.shape[0] for s in filtered_batch],
                                 'integration_domain': [[0, 100], [0, 100], [0, 50]]}
                elif mode == "val":
                    filtered_batch = [b[(b[:, 2] > 50) & (b[:, 2] <= 60)] for b in batch]
                    out_batch = {'batch_size': batch_size,
                                 'data': torch.cat(filtered_batch, dim=0),
                                 'seq_lens': [s.shape[0] for s in filtered_batch],
                                 'integration_domain': [[0, 100], [0, 100], [50, 60]]}
                elif mode == "test":
                    filtered_batch = [b[b[:, 2] > 60] for b in batch]
                    out_batch = {'batch_size': batch_size,
                                 'data': torch.cat(filtered_batch, dim=0),
                                 'seq_lens': [s.shape[0] for s in filtered_batch],
                                 'integration_domain': [[0, 100], [0, 100], [60, 100]]}

                else:
                    raise NotImplementedError
                #out_batch = {'batch_size': batch_size, 'data': torch.stack(batch)}
            return out_batch
        return collate_fn


    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, collate_fn=self.collate_fn_generator(mode="train"), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=len(self.data), collate_fn=self.collate_fn_generator(mode="val"), shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=len(self.data), collate_fn=self.collate_fn_generator(mode="test"), shuffle=True)
