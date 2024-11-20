import torch
from torch import nn
import pytorch_lightning as pl
import torch.autograd as autograd
import math
import torchquad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.math_utils import rbf_kernel,linear_kernel,matern_kernel,denoising_score_matching,score_matching,feature_extractor,f1,rq_kernel,f1_two_layer
import pandas as pd
import numpy as np
from model.process_rp import *
import scipy.stats

class DKMPP(pl.LightningModule):
    def __init__(self,
                 d=1, d_t=32, lr=1e-2, l2=1e-6, num_samples=5000,
                 loss_type='sm', kernel_type = 'dkf',
                 base_kernel = 'rbf', dsm_sigma=0.1, dataset ='synthetic'):
        super().__init__()
        self.lr = lr
        self.l2 = l2
        self.loss_type = loss_type
        self.kernel_type = kernel_type
        self.num_samples = num_samples
        self.dsm_sigma = dsm_sigma
        self.base_kernel = base_kernel
        self.dataset = dataset
        self.test_outputs = []

        rp = torch.cartesian_prod(
            torch.arange(0, 125, 25), # x
            torch.arange(0, 125, 25), # y,
            torch.arange(0, 125, 25), # t
        )
        if self.dataset == 'synthetic':
            self.z_func = lambda p: (torch.tensor((scipy.stats.norm(50, 100).pdf(p[:,0]) + scipy.stats.norm(50, 200).pdf(p[:,1])) * 10000, dtype=torch.float32))
            self.register_buffer("representative_points", torch.cat([rp, self.z_func(rp).unsqueeze(1)], dim=1))

        else:
            self.register_buffer("representative_points",concate_z(rp,self.dataset) )
        
        self.z_dim = self.representative_points[:,3:].shape[1]
        # elif self.dataset == 'vancouver':
        #     self.register_buffer("representative_points",concate_z(rp) )

        # elif self.dataset == 'collision':
        #     self.register_buffer("representative_points",concate_z(rp) )

        #===============TEST F1=======================
        torch.manual_seed(88)
        if self.dataset == "collision" or self.dataset == 'compliants':
            self.weights_raw_1 =  nn.Parameter(torch.randn(self.z_dim, 20), requires_grad=True) # 10 input features, 20 hidden units
            self.bias_raw_1 = nn.Parameter(torch.randn(20), requires_grad=True )    # 20 biases for the hidden units
            self.weights_raw_2 = nn.Parameter(torch.randn(20, 1) ,requires_grad=True)  # 20 hidden units, 1 output target
            self.bias_raw_2 = nn.Parameter(torch.randn(1) ,requires_grad=True )
            # self.weights_raw = nn.Parameter(torch.rand([self.z_dim, 1])*50, requires_grad=True)
            # self.bias_raw = nn.Parameter(torch.rand([1, 1])*100, requires_grad=True)

        else:
            self.weights_raw = nn.Parameter(torch.rand([1, 1]), requires_grad=True)
            self.bias_raw = nn.Parameter(torch.rand([1, 1]), requires_grad=True)
        #===============TEST F1=======================

        # ***************TEST WITHOUT COVARIATES************************************
        # self.weights_kernel_1 = nn.Parameter(torch.diag(torch.ones([3, ])), requires_grad=True)
        # self.weights_kernel_2 = nn.Parameter(torch.diag(torch.rand([3, ])), requires_grad=True)

        # k_theta(u,s) without covariates
        self.weights_kernel_1 = nn.Parameter(torch.rand(3,3), requires_grad=True)
        self.weights_kernel_2 = nn.Parameter(torch.rand(3,3), requires_grad=True)
        self.bias_kernel_1 = nn.Parameter(torch.ones([1, 3]), requires_grad=True)
        self.bias_kernel_2 = nn.Parameter(torch.rand([1, 3]), requires_grad=True)

        self.register_buffer('position_vec', torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_t) for i in range(d_t)]))

        self.lamda = 0.
        self.lamb_reg = 0.
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.gamma = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.mc = torchquad.MonteCarlo()
        self.l1_reg = True

    def intensity(self, x):
        if self.kernel_type == 'rbf':
            # x: [1xnbx4] rp:[125x4]
            k_x = rbf_kernel(x,self.representative_points,self.gamma) # 1xnbx125

        elif self.kernel_type == 'linear':
            k_x = linear_kernel(x,self.representative_points,self.gamma)

        elif self.kernel_type == 'matern':
            k_x = matern_kernel(x,self.representative_points,self.gamma)

        elif self.kernel_type == 'rq':
            k_x = rq_kernel(x,self.representative_points)


        elif self.kernel_type == 'dkf':
            c = feature_extractor(x[:,:,:3],self.weights_kernel_1,self.bias_kernel_1,self.weights_kernel_2,self.bias_kernel_2) # 1xnbx1
            d = feature_extractor(self.representative_points[:,:3],self.weights_kernel_1,self.bias_kernel_1, self.weights_kernel_2,self.bias_kernel_2) # 125x1

            if self.base_kernel == 'rbf':
                k_x = rbf_kernel(c,d,self.gamma)
            elif self.base_kernel == 'linear':
                k_x = linear_kernel(c,d,self.gamma)
            elif self.base_kernel == 'matern':
                k_x = matern_kernel(c,d,self.gamma)
            elif self.base_kernel == 'rq':
                k_x = rq_kernel(c,d)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


        if self.kernel_type == 'dkf':
            # use zu only
            if self.dataset == 'vancouver' or self.dataset == 'synthetic':
                f1_value = f1(self.representative_points[:,-1].unsqueeze(-1),self.weights_raw,self.bias_raw)
            else:
                #f1_value = f1(self.representative_points[:,3:],self.weights_raw,self.bias_raw)
                f1_value = f1_two_layer(self.representative_points[:,3:], 
                                        self.weights_raw_1, 
                                        self.bias_raw_1, 
                                        self.weights_raw_2, 
                                        self.bias_raw_2)

        else:
            f1_value = torch.ones(125,1)

        intensity_value = (f1_value.view(1,1,-1) * k_x).sum(2)
        #intensity_value = self.softplus((f1_value.view(1,1,-1) * k_x).sum(2))

        return intensity_value

    def gt_intensity(self,x):
        # TEST FOR SYNTHETIC DATA ONLY -- GROUND TRUTH INTENSITY OF SYNTHETIC DATA
        w1 =torch.diag(torch.from_numpy(np.array([0.99,0.98,0.97]))).float()
        b1 = (torch.ones([1, 3])*0.1).float()

        c = feature_extractor(x[:,:,:3],w1,b1,self.weights_kernel_2,self.bias_kernel_2) # 1xnbx1
        d = feature_extractor(self.representative_points[:,:3],w1,b1, self.weights_kernel_2,self.bias_kernel_2) # 125x1
        k_x = rbf_kernel(c,d,torch.tensor(0.1)) # 1xnbx125

        w2 = (torch.ones([1, 1])*0.5).float()
        b2 = (torch.ones([1, 1])*0.098).float()

        f1_value = f1(self.representative_points[:,-1].unsqueeze(-1),w2,b2)
        intensity_value = self.softplus((f1_value.view(1,1,-1) * k_x).sum(2))

        return intensity_value

    def forward(self, x):
        return self.intensity(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5),
                "monitor": 'val_loss_epoch'
            }
        }


    def loss(self, x, train):
        batch_size = x['batch_size']
        samples = x['data'].unsqueeze(0)
        integration_domain = x['integration_domain']

        loss = None
        if self.loss_type == 'sm':
            samples.requires_grad_(True)
            log_intensity = torch.log(self.intensity(samples))
            loss = score_matching(samples,log_intensity,batch_size,self.lamda,train=train)

        elif self.loss_type == 'dsm':
            v = torch.empty(samples.shape, dtype=samples.dtype,device=self.device).normal_(mean=0,std=self.dsm_sigma)
            samples_noise = samples+v
            samples_noise.requires_grad_(True)
            log_intensity = torch.log(self.intensity(samples_noise))
            loss = denoising_score_matching(samples,samples_noise,log_intensity,batch_size,self.dsm_sigma,train=train)

        elif self.loss_type == 'll':
            loss = self.log_likelihood(samples, batch_size, integration_domain)
        else:
            raise NotImplementedError

        if self.l1_reg:
            l1_reg = torch.norm(self.weights_kernel_1, 1)
            loss += self.lamb_reg * l1_reg

        if not train:
            loss = loss.detach()

        return loss

    def compute_integral_unbiased(self, integration_domain=[[0, 100], [0, 100], [0, 100]]):
        # integration domain has to be 0-1 to force volume =1 for
        # mc integration
        max_default_dim = 3
        def func_to_integral(x):
            if self.dataset == 'synthetic':
                z = self.z_func(x[:, :max_default_dim])
                z = z.to(self.device)
                x = x.to(self.device)
                # x *= 100
                x = torch.cat([x[:, :max_default_dim], z.unsqueeze(1)], dim=1)
            else:
                # x *= 100
                x = concate_z(x,self.dataset)
            return self.intensity(x.unsqueeze(0)).squeeze(0)

        return self.mc.integrate(
            func_to_integral,
            dim=3,
            N=self.num_samples,
            integration_domain=integration_domain,
            backend="torch",
            seed=88
        )

    def log_likelihood(self, samples, batch_size, integration_domain):
        all_lambda = self.intensity(samples)
        event_ll = torch.sum(torch.log(all_lambda), dim=-1)
        non_event_ll = self.compute_integral_unbiased(integration_domain)
        loss = non_event_ll - event_ll/batch_size
        print('here2', event_ll)
        # import ipdb; ipdb.set_trace()
        return loss

    def absolute_prediction_error(self, integration_domain, seq_lens):
        # predicted number of events compared to actual
        # number of events in test data
        predicted_event_number = self.compute_integral_unbiased(integration_domain)
        print('here', predicted_event_number)
        actual_num = torch.tensor(seq_lens).to(self.device)

        # actual_nonzero = torch.nonzero(actual_num)

        # print(torch.mean(torch.abs(actual_nonzero- predicted_event_number)))
        # print(torch.mean(torch.abs(actual_num- predicted_event_number)))
        
        return torch.mean(torch.abs(actual_num- predicted_event_number))
    
    def absolute_prediction_error_percentage(self, integration_domain, seq_lens):
        predicted_event_number = self.compute_integral_unbiased(integration_domain)
        # filter 0 number 
        actual_num = torch.tensor(seq_lens).to(self.device)
        # print(actual_num)
        # print(predicted_event_number)
        indices = torch.nonzero(actual_num)
        actual_non_zero = torch.index_select(actual_num, dim=0, index=indices.squeeze())


        abs_diff = torch.abs(predicted_event_number- actual_non_zero)
        p = abs_diff / (actual_non_zero)
   
        return torch.mean(p)


    def training_step(self, train_batch: torch.Tensor, batch_idx):
        loss = self.loss(train_batch, train=True)

        loss = torch.sum(loss)
        self.log('train_loss_epoch', loss, on_epoch=True, on_step=False)
        return loss

    @torch.enable_grad()
    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        loss = self.loss(val_batch, train=False)
        loss = torch.sum(loss)
        self.log('val_loss_epoch', loss, on_epoch=True, on_step=False)
        return loss

    def test_step(self, test_batch: torch.Tensor, batch_idx):
        ll = -self.log_likelihood(test_batch['data'].unsqueeze(0), test_batch['batch_size'], test_batch['integration_domain'])
        acc = self.absolute_prediction_error(test_batch['integration_domain'], test_batch['seq_lens'])
        acc_p = self.absolute_prediction_error_percentage(test_batch['integration_domain'], test_batch['seq_lens'])
        #gt_points = torch.tensor(test_batch['seq_lens']).to(self.device)

        #self.log("# of groud truth points [90,100]", gt_points.mean(),on_epoch=True,on_step=False)
        self.log('non_event_ll', self.compute_integral_unbiased())
        self.log('total_test_ll', ll, on_epoch=True, on_step=False)
        self.log('absolute prediction error', acc, on_epoch=True, on_step=False)
        self.log('diff percentage',acc_p)
        return ll
