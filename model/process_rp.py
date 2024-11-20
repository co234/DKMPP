import torch
import pandas as pd
import numpy as np
import pickle



def set_up(dataset):
    if dataset == 'vancouver':
        df = pd.read_csv("data/real_data/vancouver_crime.csv.gz")
        data = torch.tensor(np.array(df),dtype=torch.float32)
        space = data[:,:-2]
        z = data[:,3]

    elif dataset == 'collision':
        data = pickle.load(open("data/real_data/nypd_collision.pkl","rb"))
        space = torch.stack([torch.tensor(data['x'], dtype=torch.float32), 
                             torch.tensor(data['y'], dtype=torch.float32), 
                             torch.tensor(data['t'], dtype=torch.float32)], dim=1)
        z = torch.tensor(data['z'],dtype = torch.float32)

    elif dataset == 'compliants':
        data = pickle.load(open("data/real_data/nypd_compliants.pkl","rb"))
        space = torch.stack([torch.tensor(data['x'], dtype=torch.float32), 
                             torch.tensor(data['y'], dtype=torch.float32), 
                             torch.tensor(data['t'], dtype=torch.float32)], dim=1)
        z = torch.tensor(data['z'],dtype = torch.float32)

    else:
        raise NotImplementedError

    return space,z




def z_func(space,z_gt,rp):
    distance = torch.sqrt(torch.sum(torch.pow(space - rp, 2), dim=1))

    idx = int(distance.argmin().numpy())
    z = z_gt[idx]
    return z



def concate_z(rp, dataset):

    space_value, z_value = set_up(dataset)

    z_test = [z_func(space_value,z_value,rp[i]) for i in range(len(rp))]
    if dataset == 'vancouver':
        z_ = torch.stack(z_test).unsqueeze(1)
    else:
        z_ = torch.stack(z_test).squeeze(0)

    return torch.cat([rp, z_], dim=1)



