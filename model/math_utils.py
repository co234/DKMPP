import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchmetrics.functional import mean_squared_error
import numpy as np


def softplus(x, beta):
    """
    * [Eq 7] replace log in the intensity function 
    x : mxn
    return: mxn
    """
    # hard thresholding at 20
    temp = beta * x
    # temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def sp(x):
    return torch.log(1+torch.exp(x))



def denoising_score_matching(samples,samples_noise,log_intensity,batch_size,sigma,train=False):

    """
    Implement Eq (14) in the paper
    """

    # # add noise from gaussian 
    # v = torch.empty(samples.shape, dtype=samples.dtype).normal_(mean=0,std=sigma)
    # samples_noise = samples+v 

    # Compute delta[logp(s_m_tilder|s_m)]/delta[s_m_n_tilder]
    # which is (s_m_n_tilder - s_m_n)/sigma^2 
    loss_1 = (samples - samples_noise)/sigma**2
    # Compute delta[logp(s_m_tilder)]/delta[s_m_n_tilder]
    loss_2 = autograd.grad(log_intensity.sum(), samples_noise, create_graph = True, retain_graph=True)[0]

    # compute eq 14
    loss = torch.sum((loss_1-loss_2)**2)/(2*batch_size)

    return loss 


def score_matching(samples,log_intensity,batch_size,lam = 0, train=False):
    """
    Implement Eq (10) in the paper
    """

    # compute delta[logp/delta s_m_n]^2
    grad1 = autograd.grad(log_intensity.sum(), samples, create_graph=True, retain_graph=True)[0]
    # loss1 = ((grad1.sum(2) - x_intensity) ** 2 / 2)
    loss1 = ((grad1.sum(2)) ** 2 / 2)

    # compute second order derivatives
    grad2 = torch.zeros(samples.shape)
    for z in range(samples.shape[2]):
        grad = autograd.grad(grad1[:,:,z].sum(), samples, create_graph=train, retain_graph=True)[0]
        if not train:
            grad = grad.detach()
        grad2[:,:,z] = grad[:,:,z]

    loss2 = grad2.sum(2)
    loss = (loss1.sum() + loss2.sum() + lam*(loss2**2).sum())/batch_size

    return loss


def rbf_kernel(x,representative_points,gamma):
    diff = (x.unsqueeze(2) - representative_points.unsqueeze(0).unsqueeze(0))
    K = torch.exp(-(gamma**2)*torch.sum((diff**2), dim=3))

    return K 


def linear_kernel(x,representative_points,gamma):

    return x.matmul(representative_points.t())+gamma


def matern_kernel(x,representative_points,gamma):
    diff = torch.abs(x.unsqueeze(2) - representative_points.unsqueeze(0).unsqueeze(0))
    K = torch.exp(-(gamma)*torch.sum((diff), dim=3))

    return K


def rq_kernel(x,representative_points):
    diff = torch.abs(x.unsqueeze(2) - representative_points.unsqueeze(0).unsqueeze(0))
    K = 1/(1+torch.sum((diff**2), dim=3))**0.5

    return K


def f1_two_layer(x, weights1, bias1, weights2, bias2):
    # First layer
    hidden = torch.matmul(x, weights1) + bias1
    tanh = nn.Tanh()
    relu = nn.ReLU()
    hidden = tanh(hidden)
    #hidden = torch.tanh(hidden)
    
    # Second layer
    output = tanh(torch.matmul(hidden, weights2) + bias2)
    
    return output


# def feature_extractor(x,w):
#     return x.matmul(w)

def feature_extractor(X, W1, b1, W2, b2):
    # Input to hidden layer
    Z1 = X.matmul(W1) + b1
    
    # ReLU activation function
    relu = nn.ReLU()
    A1 = relu(Z1)   
    Z2 = A1.matmul(W2) + b2

    
    return Z2


def f1(X,W,b):
    Z1 = X.matmul(W) + b
    
    # ReLU activation function
    relu = nn.ReLU()
    A1 = relu(Z1)   
    
    return A1





def temporal_enc(self, x):
    x = x / self.position_vec
    new_x = torch.zeros(x.shape, device=self.device)
    new_x[:, 0::2] = torch.sin(x[:, 0::2])
    new_x[:, 1::2] = torch.cos(x[:, 1::2])
    return new_x







    






if __name__ == "__main__":
    x = torch.from_numpy(np.array([0.99,0.98,0.97,0.96]))
    w = torch.from_numpy(np.array([0.6971, 0.5620, 0.8254, 0.1217]))
    
    metric = mean_squared_error(x,w)

    print(metric)


