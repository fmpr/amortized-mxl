#!/usr/bin/env python
# coding: utf-8

# # Required imports

# In[1]:


from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from functools import partial
import logging
import math
import os
import time

import sys
RUN = int(sys.argv[1])
#RUN = 1
print("RUN number:", RUN)

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
#import biogeme.database as db
import seaborn as sns
from scipy.special import softmax
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.abstract_infer import TracePredictive
from pyro.contrib.autoguide import AutoMultivariateNormal, AutoDiagonalNormal, AutoGuideList
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

# Fix random seed for reproducibility
np.random.seed(RUN)


# In[2]:


from importlib import reload  
import logging
reload(logging)
logging.basicConfig(format='%(message)s', level=logging.INFO)


# # Generate fake data

# In[3]:


###
#Generate data
###

N = 10000
T = 10
NT = N * T
J = 5
NTJ = NT * J

L = 3 #no. of fixed paramters
K = 5 #no. of random parameters

true_alpha = np.array([-0.8, 0.8, 1.2])
true_beta = np.array([-0.8, 0.8, 1.0, -0.8, 1.5])
true_Omega = np.array([[1.0, 0.8, 0.8, 0.8, 0.8],
                       [0.8, 1.0, 0.8, 0.8, 0.8],
                       [0.8, 0.8, 1.0, 0.8, 0.8],
                       [0.8, 0.8, 0.8, 1.0, 0.8],
                       [0.8, 0.8, 0.8, 0.8, 1.0]])
# dynamic version
corr = 0.8
scale_factor = 1.0
true_Omega = corr*np.ones((K,K)) # off-diagonal values of cov matrix
true_Omega[np.arange(K), np.arange(K)] = 1.0 # diagonal values of cov matrix
true_Omega *= scale_factor

print("Generating fake data...")
xFix = np.random.rand(NTJ, L)
xRnd = np.random.rand(NTJ, K)

betaInd_tmp = true_beta + (np.linalg.cholesky(true_Omega) @ np.random.randn(K, N)).T
beta_tmp = np.kron(betaInd_tmp, np.ones((T * J,1)))

eps = -np.log(-np.log(np.random.rand(NTJ,)))

vDet = xFix @ true_alpha + np.sum(xRnd * beta_tmp, axis = 1)
v = vDet + eps

vDetMax = np.zeros((NT,))
vMax = np.zeros((NT,))

chosen = np.zeros((NTJ,), dtype = 'int64')

for t in np.arange(NT):
    l = t * J; u = (t + 1) * J
    altMaxDet = np.argmax(vDet[l:u])
    altMax = np.argmax(v[l:u])
    vDetMax[t] = altMaxDet
    vMax[t] = altMax
    chosen[l + altMax] = 1

error = np.sum(vMax == vDetMax) / NT * 100
print("Error:", error)

indID = np.repeat(np.arange(N), T * J)
obsID = np.repeat(np.arange(NT), J)
altID = np.tile(np.arange(J), NT)  


# ### Convert fake data to wide format

# In[4]:


num_alternatives = altID.max() + 1
num_resp = indID.max() + 1


# In[5]:


if True: # THIS IS SLOW!!! IF NOT CHANGED, IT IS FASTER TO READ THE PREVIOUS DATA FROM DISK
    # convert long format to wide format
    xs = []
    ys = []
    for ind in range(num_resp):
        #print("------------------ individual:", ind)
        ind_ix = np.where(indID == ind)[0]
        #print("ind_ix:", ind_ix)
        ind_xs = []
        ind_ys = []
        for n in np.unique(obsID[ind_ix]):
            #print("--------- observation:", n)
            obs_ix = np.where(obsID == n)[0]
            #print("obs_ix:", obs_ix)

            # get attributes (x)
            x = [[] for i in range(num_alternatives)]
            #print("altID:", altID[obs_ix])
            for alt in range(num_alternatives):
                if alt in altID[obs_ix]:
                    x[alt].append(np.hstack([xFix[obs_ix][alt], xRnd[obs_ix][alt]]))
                else:
                    x[alt].append(np.zeros(L+K))
            x = np.hstack(x)[0]
            #print("x:", x)
            ind_xs.append(x)

            # get choice (y)
            y = np.argmax(chosen[obs_ix])
            #print("y:", y)
            ind_ys.append(y)

        xs.append(np.array(ind_xs))
        ys.append(np.array(ind_ys))

    alt_availability = np.ones((N,T,J))
    alt_attributes = np.array(xs)
    true_choices = np.array(ys)
    
    np.savez('fakedata.npz', 
             alt_availability=alt_availability, 
             alt_attributes=alt_attributes, 
             true_choices=true_choices)
else:
    # load previously generated data from disk
    data = np.load('fakedata.npz')
    alt_availability = data['alt_availability']
    alt_attributes = data['alt_attributes']
    true_choices = data['true_choices']


# In[6]:


print("Alt. availability:", alt_availability.shape)
print("Alt. attributes:", alt_attributes.shape)
print("True choices:", true_choices.shape)


# # Data preparation and Mixed Logit specification

# In[7]:


# DCM specification
num_obs = len(chosen)
print("Num. observations:", num_obs)

alt_names = ["ALT1", "ALT2", "ALT3", "ALT4", "ALT5"]
assert num_alternatives == len(alt_names)
print("Num. alternatives:", num_alternatives)

attr_names = ['ALT1_XF1', 'ALT1_XF2','ALT1_XF3', 'ALT1_XR1', 'ALT1_XR2','ALT1_XR3', 'ALT1_XR4', 'ALT1_XR5', 
              'ALT2_XF1', 'ALT2_XF2','ALT2_XF3', 'ALT2_XR1', 'ALT2_XR2','ALT2_XR3', 'ALT2_XR4', 'ALT2_XR5', 
              'ALT3_XF1', 'ALT3_XF2','ALT3_XF3', 'ALT3_XR1', 'ALT3_XR2','ALT3_XR3', 'ALT3_XR4', 'ALT3_XR5', 
              'ALT4_XF1', 'ALT4_XF2','ALT4_XF3', 'ALT4_XR1', 'ALT4_XR2','ALT4_XR3', 'ALT4_XR4', 'ALT4_XR5', 
              'ALT5_XF1', 'ALT5_XF2','ALT5_XF3', 'ALT5_XR1', 'ALT5_XR2','ALT5_XR3', 'ALT5_XR4', 'ALT5_XR5', ] 
alt_ids = np.array([0,0,0,0,0,0,0,0,
                    1,1,1,1,1,1,1,1,
                    2,2,2,2,2,2,2,2,
                    3,3,3,3,3,3,3,3,
                    4,4,4,4,4,4,4,4]) # assigns attributes to IDs corresponding to alternatives
param_ids = np.array([0,1,2,3,4,5,6,7,
                      0,1,2,3,4,5,6,7,
                      0,1,2,3,4,5,6,7,
                      0,1,2,3,4,5,6,7,
                      0,1,2,3,4,5,6,7]) # assigns attributes to IDs indicating parameters to be estimated
mix_params = np.array([3,4,5,6,7]) # IDs of parameters to be treated with a Mixed Logit formulation
non_mix_params = np.array([x for x in range(max(param_ids)+1) if x not in mix_params])
print("Parameter IDs to be treated in a Mixed Logit way:", mix_params)
print("Parameter IDs to be treated in a MNL way:", non_mix_params)

# debug utility functions specified
print("Utility functions:")
for i in range(num_alternatives):
    v_ix = np.where(alt_ids == i)[0]
    if param_ids[v_ix[0]] in mix_params:
        s = "\tV_%s_n = beta%d_n * %s" % (alt_names[i], param_ids[v_ix[0]], attr_names[v_ix[0]])
    else:
        s = "\tV_%s_n = beta%d * %s" % (alt_names[i], param_ids[v_ix[0]], attr_names[v_ix[0]])
    for j in range(1,len(v_ix)):
        if param_ids[v_ix[j]] in mix_params:
            s += " + beta%d_n * %s" % (param_ids[v_ix[j]], attr_names[v_ix[j]])
        else:
            s += " + beta%d * %s" % (param_ids[v_ix[j]], attr_names[v_ix[j]])
    print(s)

# further checks and definitions
assert len(np.unique(param_ids)) == max(param_ids)+1
assert min(param_ids) == 0
num_params = max(param_ids) + 1
print("Num. parameters to be estimated:", num_params)
D = len(attr_names)
print("Num. attributes to be used in total:", D)
assert len(attr_names) == len(alt_ids) # length check
assert max(alt_ids) + 1 == num_alternatives    

resp_ids = np.arange(num_resp)
print("Num respondents:", num_resp)


# # Bayesian Mixed Logit Model

# In[8]:


# auxiliary dictionary for Pyro model implementation
beta_to_params_map = [param_ids[np.where(alt_ids == i)[0]] for i in range(num_alternatives)]

# auxiliary CUDA matrix for Pyro model
zeros_vec = torch.zeros(T,num_resp,num_alternatives).cuda()

pyro.enable_validation(True)    # <---- This is always a good idea!


# In[9]:


#BATCH_SIZE = num_resp 
BATCH_SIZE = 2000 # CHANGED
#BATCH_SIZE = int(num_resp / 5)
print("Batch size:", BATCH_SIZE)

diagonal_alpha = False
diagonal_beta_mu = False

def model(x, y, alt_av, alt_ids_cuda):
    # global parameters in the model
    if diagonal_alpha:
        alpha_mu = pyro.sample("alpha", dist.Normal(torch.zeros(len(non_mix_params), device=x.device), 1).to_event(1))
    else:
        alpha_mu = pyro.sample("alpha", dist.MultivariateNormal(torch.zeros(len(non_mix_params), device=x.device), 
                                            scale_tril=torch.tril(1*torch.eye(len(non_mix_params), device=x.device))))
    
    if diagonal_beta_mu:
        beta_mu = pyro.sample("beta_mu", dist.Normal(torch.zeros(len(mix_params), device=x.device), 1.).to_event(1))
    else:
        beta_mu = pyro.sample("beta_mu", dist.MultivariateNormal(torch.zeros(len(mix_params), device=x.device), 
                                            scale_tril=torch.tril(1*torch.eye(len(mix_params), device=x.device))))
    
    # Vector of variances for each of the d variables
    theta = pyro.sample("theta", dist.HalfCauchy(10.*torch.ones(len(mix_params), device=x.device)).to_event(1))
    # Lower cholesky factor of a correlation matrix
    eta = 1.*torch.ones(1, device=x.device)  # Implies a uniform distribution over correlation matrices
    L_omega = pyro.sample("L_omega", dist.LKJCorrCholesky(len(mix_params), eta))
    # Lower cholesky factor of the covariance matrix
    L_Omega = torch.mm(torch.diag(theta.sqrt()), L_omega)
        
    # local parameters in the model
    random_params = pyro.sample("beta_resp", dist.MultivariateNormal(beta_mu.repeat(num_resp,1), 
                                                                     scale_tril=L_Omega).to_event(1))
    
    # vector of respondent parameters: global + local (respondent)
    params_resp = torch.cat([alpha_mu.repeat(num_resp,1), random_params], dim=-1)

    # vector of betas of MXL (may repeat the same learnable parameter multiple times; random + fixed effects)
    beta_resp = torch.cat([params_resp[:,beta_to_params_map[i]] for i in range(num_alternatives)], dim=-1)
    
    with pyro.plate("locals", len(x), subsample_size=BATCH_SIZE) as ind:
        
        with pyro.plate("data_resp", T):
            # compute utilities for each alternative
            utilities = torch.scatter_add(zeros_vec[:,ind,:],
                                          2, 
                                          alt_ids_cuda[ind,:,:].transpose(0,1), 
                                          torch.mul(x[ind,:,:].transpose(0,1), beta_resp[ind,:]))
            
            # adjust utility for unavailable alternatives
            utilities += alt_av[ind,:,:].transpose(0,1)

            # likelihood
            pyro.sample("obs", dist.Categorical(logits=utilities), obs=y[ind,:].transpose(0,1))
            


# # Specify variational approximation q (guide)

# In[10]:


from torch import nn

kernel_size = num_params*num_alternatives+num_alternatives*2
print("Kernel size:", kernel_size)

class Predictor(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.z_dim = z_dim
        self.cnn1 = torch.nn.Conv1d(1, 200, kernel_size=(kernel_size), stride=(kernel_size), 
                                    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.cnn2 = torch.nn.Conv1d(200, 200, kernel_size=(1), stride=(1), 
                                    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.bn = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc1 = nn.Linear(200, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3mu = nn.Linear(hidden_dim, z_dim)
        self.fc3sigma = nn.Linear(hidden_dim, z_dim)
        self.fc3offdiag = nn.Linear(hidden_dim, int((z_dim*(z_dim-1))/2))
        
        self.dropout = nn.Dropout(0.5)
        #self.pooling = nn.AvgPool1d(T, stride=(T))
        self.pooling = nn.MaxPool1d(T, stride=(T))
        
        self.tril_indices = torch.tril_indices(row=z_dim, col=z_dim, offset=-1).cuda()
        
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, x):
        # compute the hidden units
        hidden = self.bn(x)
        #hidden = x
        hidden = self.cnn1(hidden)
        hidden = self.relu(self.pooling(hidden))
        hidden = self.bn2(hidden)
        #hidden = hidden.flatten(1,2)
        hidden = self.relu(self.fc1(hidden.flatten(1,2)))
        
        # return a mean vector (batch_size x z_dim) and a lower-triangular Cholesky factor of a MVN
        z_loc = self.fc3mu(hidden)
        z_diag = self.softplus(self.fc3sigma(hidden))
        z_offdiag = self.fc3offdiag(hidden)
        
        z_tril = torch.zeros((x.shape[0], self.z_dim, self.z_dim), device=x.device)
        z_tril[:, self.tril_indices[0], self.tril_indices[1]] = z_offdiag
        z_tril += torch.diag_embed(z_diag)

        return z_loc, z_tril
    
predictor = Predictor(K, 200).cuda()
print(predictor)


# In[11]:


layers = [predictor.cnn1, predictor.fc1, predictor.bn, predictor.bn2, 
          predictor.fc3mu, predictor.fc3sigma, predictor.fc3offdiag]
pytorch_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in layers]
print("Parameters by layer:", pytorch_params)
print("Total parameters:", sum(pytorch_params))


# In[12]:


alt_av_cuda = torch.from_numpy(alt_availability)
alt_av_cuda = alt_av_cuda.cuda()


# In[13]:


from torch.nn.functional import softplus

def my_local_guide(x, y, alt_av, alt_ids):
    if diagonal_alpha:
        alpha_loc = pyro.param('alpha_loc', torch.randn(len(non_mix_params), device=x.device))
        alpha_scale = pyro.param('alpha_scale', 1*torch.ones(len(non_mix_params), device=x.device),
                                 constraint=constraints.positive)
        alpha = pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).to_event(1))
    else:
        alpha_loc = pyro.param('alpha_loc', torch.randn(len(non_mix_params), device=x.device))
        alpha_scale = pyro.param("alpha_scale", torch.tril(1*torch.eye(len(non_mix_params), device=x.device)),
                                 constraint=constraints.lower_cholesky)
        alpha = pyro.sample("alpha", dist.MultivariateNormal(alpha_loc, scale_tril=alpha_scale))
    
    if diagonal_beta_mu:
        beta_mu_loc = pyro.param('beta_mu_loc', torch.randn(len(mix_params), device=x.device))
        beta_mu_scale = pyro.param('beta_mu_scale', 1*torch.ones(len(mix_params), device=x.device),
                                   constraint=constraints.positive)
        beta_mu = pyro.sample("beta_mu", dist.Normal(beta_mu_loc, beta_mu_scale).to_event(1))
    else:
        beta_mu_loc = pyro.param('beta_mu_loc', torch.randn(len(mix_params), device=x.device))
        beta_mu_scale = pyro.param("beta_mu_scale", torch.tril(1*torch.eye(len(mix_params), device=x.device)),
                                   constraint=constraints.lower_cholesky)
        beta_mu = pyro.sample("beta_mu", dist.MultivariateNormal(beta_mu_loc, scale_tril=beta_mu_scale))
    
    # Use an amortized guide for local variables.
    pyro.module("predictor", predictor)
    one_hot = torch.zeros(num_resp, T, num_alternatives, device=x.device, dtype=torch.float)
    one_hot = one_hot.scatter(2, y.unsqueeze(2).long(), 1)
    inference_data = torch.cat([one_hot, x, alt_av_cuda.float()], dim=-1)
    beta_loc, beta_scale_tril = predictor.forward(inference_data.flatten(1,2).unsqueeze(1))
    pyro.sample("beta_resp", dist.MultivariateNormal(beta_loc, scale_tril=beta_scale_tril).to_event(1))
        


# In[14]:


guide = AutoGuideList(model)
guide.add(AutoDiagonalNormal(poutine.block(model, expose=['theta', 'L_omega'])))
guide.add(my_local_guide)  # automatically wrapped in an AutoCallable


# # Run variational inference

# In[15]:


# prepare data for running inference
train_x = torch.tensor(alt_attributes, dtype=torch.float)
train_x = train_x.cuda()
train_y = torch.tensor(true_choices, dtype=torch.int)
train_y = train_y.cuda()
alt_av_cuda = torch.from_numpy(alt_availability)
alt_av_cuda = alt_av_cuda.cuda()
alt_av_mat = alt_availability.copy()
alt_av_mat[np.where(alt_av_mat == 0)] = -1e9
alt_av_mat -= 1
alt_av_mat_cuda = torch.from_numpy(alt_av_mat).float()
alt_av_mat_cuda = alt_av_mat_cuda.cuda()
#alt_ids_cuda = torch.from_numpy(alt_ids[:,np.newaxis].repeat(1*num_resp,1).T.reshape(num_resp,1,-1))
alt_ids_cuda = torch.from_numpy(alt_ids[:,np.newaxis].repeat(T*num_resp,1).T.reshape(num_resp,T,-1))
alt_ids_cuda = alt_ids_cuda.cuda()


# In[16]:


one_hot = torch.zeros(num_resp, T, num_alternatives, device=train_x.device, dtype=torch.float)
one_hot = one_hot.scatter(2, train_y.unsqueeze(2).long(), 1)
inference_data = torch.cat([one_hot, train_x, alt_av_cuda.float()], dim=-1)
beta_loc, beta_scale_tril = predictor.forward(inference_data.flatten(1,2).unsqueeze(1))


# In[17]:


#trace = poutine.trace(model).get_trace(train_x, train_y, alt_av_mat_cuda, alt_ids_cuda)
#trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
#print(trace.format_shapes())


# In[18]:


from scipy.special import softmax

# function for calculating likelihood and accuracy
def loglikelihood(X, y, alt_av, alpha, beta, beta_resps):
    # gather vector of params for respondent
    params_resp = np.hstack([alpha[:,np.newaxis].repeat(num_resp,1).T, beta_resps])
    
    # build vector of betas for respondent
    beta_resp = np.hstack([params_resp[:,param_ids[np.where(alt_ids == i)[0]]] for i in range(num_alternatives)])
    
    # calculate utilities based on params
    utilities = np.zeros((num_resp, T, J))
    for resp_id in range(num_resp):
        for i in range(num_alternatives):
            utilities[resp_id,:,i] = np.dot(X[resp_id,:,np.where(alt_ids == i)[0]].T, 
                                            beta_resp[resp_id, np.where(alt_ids == i)[0]]).T

    # adjust utility for unavailable alternatives
    utilities += alt_av

    # likelihood
    probs = softmax(utilities, axis=2)
    loglik = np.sum(np.log(probs.reshape(num_resp*T,J)[np.arange(num_resp*T), y.flatten()]))
    acc = np.mean(np.argmax(probs, axis=2) == y[:,:])
    
    return loglik, acc

def sim_loglikelihood(X, y, alt_av, alpha, beta, betaCovChol, num_samples=1000):
    #betaCovChol = np.linalg.cholesky(betaCov)
    pSim = np.zeros((num_samples, num_resp))

    for i in np.arange(num_samples):
        paramRnd = beta + (betaCovChol @ np.random.randn(K, num_resp)).T

        # gather vector of params for respondent
        params_resp = np.hstack([alpha[:,np.newaxis].repeat(num_resp,1).T, paramRnd])

        # build vector of betas for respondent
        beta_resp = np.hstack([params_resp[:, param_ids[np.where(alt_ids == i)[0]]] for i in range(num_alternatives)])

        for resp_id in range(num_resp):
            # calculate utilities based on params
            utilities = np.vstack([np.dot(X[resp_id,:,np.where(alt_ids == i)[0]].T, 
                                          beta_resp[resp_id, np.where(alt_ids == i)[0]]) for i in range(num_alternatives)])

            # adjust utility for unavailable alternatives
            utilities = utilities.T + alt_av[resp_id]

            # likelihood
            probs = softmax(utilities, axis=1)
            lPInd = np.sum(np.log(probs[np.arange(T), y[resp_id]]))

            pSim[i, resp_id] = np.exp(lPInd)

    logLik = np.sum(np.log(np.mean(pSim, axis=0)))
    
    return logLik


# In[19]:


def per_param_args(module_name, param_name):
    if 'predictor' in module_name:
        return {"lr": 0.0002} 
    elif '_loc' in param_name:
        return {"lr": 0.005}
    elif '_scale' in param_name:
        return {"lr": 0.005} # CHANGED
    else:
        return {"lr": 0.005}


# In[20]:


svi = SVI(model,
          guide,
          #optim.ClippedAdam({"lr": 0.005}),
          optim.ClippedAdam(per_param_args),
          loss=Trace_ELBO(),
          num_samples=1000)
pyro.clear_param_store()
    
num_epochs = 10000
track_loglik = True
elbo_losses = []
alpha_errors = []
beta_errors = []
betaInd_errors = []
best_elbo = np.inf
patience_thre = 3
patience_count = 0
tic = time.time()
for j in range(num_epochs):
    elbo = svi.step(train_x, train_y, alt_av_mat_cuda, alt_ids_cuda)
    elbo_losses += [elbo]
    
    if j % 100 == 0:
        if track_loglik:
            alpha_params = pyro.param("alpha_loc").data.cpu().numpy()
            beta_params = pyro.param("beta_mu_loc").data.cpu().numpy()
            
            beta_loc, beta_scale_tril = predictor.forward(inference_data.flatten(1,2).unsqueeze(1))
            params_resps = beta_loc.detach().cpu().numpy()
            
            alpha_rmse = np.sqrt(np.mean((true_alpha - alpha_params)**2))
            beta_rmse = np.sqrt(np.mean((true_beta - beta_params)**2))
            params_resps_rmse = np.sqrt(np.mean((betaInd_tmp - params_resps)**2))
            alpha_errors += [alpha_rmse]
            beta_errors += [beta_rmse]
            betaInd_errors += [params_resps_rmse]
            
            loglik, acc = loglikelihood(alt_attributes, true_choices, alt_av_mat, 
                                        alpha_params, beta_params, params_resps)
            logging.info("[Epoch %d] Elbo: %.0f; Loglik: %.0f; Acc.: %.3f; Alpha RMSE: %.3f; Beta RMSE: %.3f; BetaInd RMSE: %.3f" % (j, 
                                                                          elbo, loglik, acc, alpha_rmse, beta_rmse, params_resps_rmse))
        else:
            logging.info("Elbo loss: %.2f" % (elbo,))
            
        if np.mean(elbo_losses[-500::10]) < best_elbo:
            best_elbo = np.mean(elbo_losses[-500::10])
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience_thre:
                logging.info("Elbo converged!")
                break
            
toc = time.time() - tic
print("Elapsed time:", toc)


# In[21]:


np.set_printoptions(precision=3)

alpha_params = pyro.param("alpha_loc").data.cpu().numpy()
beta_params = pyro.param("beta_mu_loc").data.cpu().numpy()
beta_params_cov = pyro.param("beta_mu_scale").data.cpu().numpy()
            
beta_loc, beta_scale_tril = predictor.forward(inference_data.flatten(1,2).unsqueeze(1))
params_resps = beta_loc.detach().cpu().numpy()

alpha_error = np.abs(true_alpha - alpha_params).mean()
alpha_rmse = np.sqrt(np.mean((true_alpha - alpha_params)**2))
beta_error = np.abs(true_beta - beta_params).mean()
beta_rmse = np.sqrt(np.mean((true_beta - beta_params)**2))
params_resps_error = np.abs(betaInd_tmp - params_resps).mean()
params_resps_rmse = np.sqrt(np.mean((betaInd_tmp - params_resps)**2))

loglik, acc = loglikelihood(alt_attributes, true_choices, alt_av_mat, 
                            alpha_params, beta_params, params_resps)

loglik_hyp,_ = loglikelihood(alt_attributes, true_choices, alt_av_mat, 
                             alpha_params, beta_params, np.tile(beta_params, [N,T]))


# In[22]:


try:
    svi_posterior = svi.run(train_x, train_y, alt_av_mat_cuda, alt_ids_cuda)
except:
    pass


# In[23]:


L_omega_posterior = EmpiricalMarginal(svi, sites=["L_omega"])._get_samples_and_weights()[0]
L_omega = L_omega_posterior.mean(axis=0)[0].detach().cpu().numpy()
theta_posterior = EmpiricalMarginal(svi, sites=["theta"])._get_samples_and_weights()[0]
L_Omega = torch.mm(torch.diag(theta_posterior.mean(axis=0)[0].sqrt()), L_omega_posterior.mean(axis=0)[0])
L_Omega = L_Omega.detach().cpu().numpy()

Omega_params = np.dot(L_Omega,L_Omega.T)
Omega_rmse = np.sqrt(np.mean((true_Omega - Omega_params)**2))


# In[24]:


print("Num. posterior samples:", L_omega_posterior.shape[0])
print()


# In[25]:


print("True alpha:", true_alpha)
print("Estimated alpha:", alpha_params)
print("Mean error (alpha):", alpha_error)
print("RMSE (alpha):", alpha_rmse)
print("\nTrue beta:", true_beta)
print("Estimated beta:", beta_params)
print("Mean error (beta):", beta_error)
print("RMSE (beta):", beta_rmse)
print("\nTrue Omega:", true_Omega)
print("Estimated Omega:", Omega_params)
print("RMSE (Omega):", Omega_rmse)
print("\nMean error (params resps):", params_resps_error)
print("RMSE (params resps):", params_resps_rmse)
print("\nLoglikelihood:", loglik)
print("\nLoglikelihood (hyper-priors only):", loglik_hyp)

sim_loglik = sim_loglikelihood(alt_attributes, true_choices, alt_av_mat, 
                               alpha_params, beta_params, L_Omega, num_samples=200)
print("\nLoglikelihood (simulated at posterior means):", sim_loglik)


# In[29]:


output_dir = "Results_FakeData_N%d_T%d_J%d_L%d_K%d_Corr%.1f_Scale%.1f_Batch%d" % (N,T,J,L,K,
                                                                                   corr,scale_factor,
                                                                                   BATCH_SIZE)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fname = output_dir + "/Pyro_LKJ_AmortizedVI_v2d.txt"
if not os.path.exists(fname):
    fw = open(fname, "w")
    fw.write("Run\tTime\tLoglik\tSim. Loglik\tLoglik (hyper)\tRMSE alpha\tRMSE beta\tRMSE betaInd\tRMSE Omega\n")
else:
    fw = open(fname, "a")
    
fw.write("%d\t%.0f\t%.1f\t%.1f\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (RUN, toc, 
                                                            loglik, sim_loglik, loglik_hyp, 
                                                            alpha_rmse, beta_rmse, params_resps_rmse, Omega_rmse))
fw.close()


# In[30]:


import pickle
with open(fname.replace(".txt","_Run%d.pickle" % (RUN,)), 'wb') as f:
    pickle.dump({"elbo_losses": elbo_losses,
                 "alpha_errors": alpha_errors,
                 "beta_errors": beta_errors,
                 "betaInd_errors": betaInd_errors}, 
                f, protocol=pickle.HIGHEST_PROTOCOL)

