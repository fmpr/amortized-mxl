#!/usr/bin/env python
# coding: utf-8

# # Required imports

# In[1]:


import sys
sys.path.append("..") # Adds higher directory to python modules path.

#from joblib import Parallel, delayed
import os
import time
import pandas as pd
import numpy as np
from scipy.stats import invwishart
import scipy.sparse
from math import floor
import h5py

from mxl import corrcov, prepareData, mvnlpdf, probMxl, pPredMxl


# # Bayesian Mixed Logit Model
# 
# Generative process:
# 
# 1. Draw fixed taste parameters $\boldsymbol\alpha | \boldsymbol\lambda_0, \boldsymbol\Xi_0 \sim \mathcal{N}(\boldsymbol\lambda_0, \boldsymbol\Xi_0)$
# 
# <!-- sep -->
# 
# 2. Draw prior mean vector $\boldsymbol\zeta | \boldsymbol\mu_0, \boldsymbol\Sigma_0 \sim \mathcal{N}(\boldsymbol\mu_0, \boldsymbol\Sigma_0)$
# 
# 3. Draw hyper-prior $a_k | A_k \sim \mbox{Gamma}\big(\frac{1}{2}, \frac{1}{A_k^2}\big)$ for $k=1,\dots,K$
# 
# 4. Draw prior covariance matrix $\boldsymbol\Omega | \nu, \textbf{a} \sim \mbox{IW}\big(\nu + K - 1, 2\nu \, \mbox{diag}(\textbf{a})\big)$
# 
# <!-- sep -->
# 
# 5. For each decision-maker $n \in \{1,\dots,N\}$:
# 
#     1. Draw random taste parameters $\boldsymbol\beta_n | \boldsymbol\zeta, \boldsymbol\Omega \sim \mathcal{N}(\boldsymbol\zeta, \boldsymbol\Omega)$
#     
#     2. For each choice occasion $t \in \{1,\dots,T_n\}$:
#     
#         * Draw observed choice $y_{nt} | \boldsymbol\alpha, \boldsymbol\beta_n, \textbf{X}_{nt} \sim \mbox{MNL}(\boldsymbol\alpha, \boldsymbol\beta_n, \textbf{X}_{nt})$

# # Gibbs sampler
# 
# ### Updates
# 
# Update $\boldsymbol\zeta$ by sampling $\boldsymbol\zeta \sim \mathcal{N}\Big(\frac{1}{N} \sum_{n=1}^N \boldsymbol\beta_n, \frac{\boldsymbol\Omega}{N}\Big)$

# In[2]:


def next_zeta(paramRnd, Omega, nRnd, nInd):
    zeta = paramRnd.mean(axis = 0) + np.linalg.cholesky(Omega) @ np.random.randn(nRnd,) / np.sqrt(nInd)
    return zeta


# Update $\boldsymbol\Omega$ by sampling $\boldsymbol\Omega \sim \mbox{IW}\Big(\nu+N+K-1,2\nu \, \mbox{diag}(\textbf{a}) + \sum_{n=1}^N (\boldsymbol\beta_n-\boldsymbol\zeta)(\boldsymbol\beta_n-\boldsymbol\zeta)^T \Big)$

# In[3]:


def next_Omega(paramRnd, zeta, nu, iwDiagA, diagCov, nRnd, nInd):
    betaS = paramRnd - zeta
    Omega = np.array(invwishart.rvs(nu + nInd + nRnd - 1, 
                                    2 * nu * np.diag(iwDiagA) + betaS.T @ betaS)).reshape((nRnd, nRnd))
    if diagCov: Omega = np.diag(np.diag(Omega))
    return Omega


# Update $a_k$ for all $k \in \{1,\dots,K\}$ by sampling $a_k \sim \mbox{Gamma}\Big( \frac{\nu+K}{2}, \frac{1}{A_k^2} + \nu \, (\boldsymbol\Omega^{-1})_{kk} \Big)$

# In[4]:


def next_iwDiagA(Omega, nu, invASq, nRnd):
    iwDiagA = np.random.gamma((nu + nRnd) / 2, 1 / (invASq + nu * np.diag(np.linalg.inv(Omega))))
    return iwDiagA


# Update $\boldsymbol\beta_n$ for all $n \in \{1,\dots,N\}$:
# 
# * Propose $\tilde{\boldsymbol\beta}_n = \boldsymbol\beta_n + \sqrt{\rho_\boldsymbol\beta} \, \mbox{chol}(\boldsymbol\Omega) \, \boldsymbol\eta$, where $\boldsymbol\eta \sim \mathcal{N}(\textbf{0},\textbf{I}_K)$
# 
# 
# * Compute $r = \frac{ P(y_n|\textbf{X}_n,\boldsymbol\alpha,\tilde{\boldsymbol\beta}_n) \, \phi(\tilde{\boldsymbol\beta}_n|\boldsymbol\zeta,\boldsymbol\Omega) }{ P(y_n|\textbf{X}_n,\boldsymbol\alpha,{\boldsymbol\beta}_n) \, \phi({\boldsymbol\beta}_n|\boldsymbol\zeta,\boldsymbol\Omega) }$
# 
# 
# * Draw $u \sim \mbox{Uniform}(0,1)$. If $r \leq u$, accept the proposal, else reject it

# In[5]:


def next_paramRnd(
        paramFix, paramRnd, zeta, Omega,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rho):
    lPhi = mvnlpdf(paramRnd, zeta, Omega)
    paramRnd_star = paramRnd + np.sqrt(rho) * (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T    
    lPInd_star = probMxl(
        paramFix, paramRnd_star,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
    lPhi_star = mvnlpdf(paramRnd_star, zeta, Omega)

    r = np.exp(lPInd_star + lPhi_star - lPInd - lPhi)
    idxAccept = np.random.rand(nInd,) <= r

    paramRnd[idxAccept, :] = np.array(paramRnd_star[idxAccept, :])
    lPInd[idxAccept] = np.array(lPInd_star[idxAccept])

    acceptRate = np.mean(idxAccept)
    rho = rho - 0.001 * (acceptRate < 0.3) + 0.001 * (acceptRate > 0.3)
    return paramRnd, lPInd, rho


# Update $\boldsymbol\alpha$:
# 
# * Propose $\tilde{\boldsymbol\alpha} = \boldsymbol\alpha + \sqrt{\rho_\boldsymbol\alpha} \, \mbox{chol}(\boldsymbol\Xi_0) \, \boldsymbol\eta$, where $\boldsymbol\eta \sim \mathcal{N}(\textbf{0},\textbf{I}_L)$
# 
# 
# * Compute $r = \frac{ \prod_{n=1}^N P(y_n|\textbf{X}_n,\tilde{\boldsymbol\alpha},{\boldsymbol\beta}_n) \, \phi(\tilde{\boldsymbol\alpha}|\boldsymbol\lambda_0,\boldsymbol\Xi_0) }{ \prod_{n=1}^N P(y_n|\textbf{X}_n,\boldsymbol\alpha,{\boldsymbol\beta}_n) \, \phi({\boldsymbol\alpha}|\boldsymbol\lambda_0,\boldsymbol\Xi_0) }$
# 
# 
# * Draw $u \sim \mbox{Uniform}(0,1)$. If $r \leq u$, accept the proposal, else reject it

# In[6]:


def next_paramFix(
        paramFix, paramRnd,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rhoF):
    paramFix_star = paramFix + np.sqrt(rhoF) * np.random.randn(nFix,)
    lPInd_star = probMxl(
        paramFix_star, paramRnd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
    r = np.exp(np.sum(lPInd_star - lPInd, axis = 0))
    if np.random.rand() <= r:
        paramFix = np.array(paramFix_star)
        lPInd = np.array(lPInd_star)
    return paramFix, lPInd


# ### MCMC chain

# In[7]:


def mcmcChain(
        chainID, seed,
        mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
        rhoF, rho,
        modelName,
        paramFix, zeta, Omega, invASq, nu, diagCov,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd, 
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Precomputations
    ###
    
    if nRnd > 0:
        paramRnd = zeta + (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T
        iwDiagA = np.random.gamma(1 / 2, 1 / invASq)
    else:
        paramRnd = np.zeros((0,0))
        iwDiagA = np.zeros((0,0))
    
    lPInd = probMxl(
            paramFix, paramRnd,
            xFix, xFix_transBool, xFix_trans, nFix, 
            xRnd, xRnd_transBool, xRnd_trans, nRnd,
            nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)   
    
    ###
    #Storage
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    if os.path.exists(fileName):
        os.remove(fileName) 
    file = h5py.File(fileName, "a")
    
    if nFix > 0:
        paramFix_store = file.create_dataset('paramFix_store', (mcmc_iterSampleThin, nFix))
        
        paramFix_store_tmp = np.zeros((mcmc_iterMemThin, nFix))
        
    if nRnd > 0:
        paramRnd_store = file.create_dataset('paramRnd_store', (mcmc_iterSampleThin, nInd, nRnd))
        zeta_store = file.create_dataset('zeta_store', (mcmc_iterSampleThin, nRnd))
        Omega_store = file.create_dataset('Omega_store', (mcmc_iterSampleThin, nRnd, nRnd))
        Corr_store = file.create_dataset('Corr_store', (mcmc_iterSampleThin, nRnd, nRnd))
        sd_store = file.create_dataset('sd_store', (mcmc_iterSampleThin, nRnd))
        
        paramRnd_store_tmp = np.zeros((mcmc_iterMemThin, nInd, nRnd))
        zeta_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
        Omega_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        Corr_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        sd_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
    
    ###
    #Sample
    ###
    
    j = -1
    ll = 0
    sampleState = 'burn in'
    for i in np.arange(mcmc_iter):
        if nFix > 0:
            paramFix, lPInd = next_paramFix(
                    paramFix, paramRnd,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rhoF)
            
        if nRnd > 0:
            zeta = next_zeta(paramRnd, Omega, nRnd, nInd)
            Omega = next_Omega(paramRnd, zeta, nu, iwDiagA, diagCov, nRnd, nInd)
            iwDiagA = next_iwDiagA(Omega, nu, invASq, nRnd)
            paramRnd, lPInd, rho = next_paramRnd(
                    paramFix, paramRnd, zeta, Omega,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rho)
        
        if ((i + 1) % mcmc_disp) == 0:
            if (i + 1) > mcmc_iterBurn:
                sampleState = 'sampling'
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (' + sampleState + ')')
            sys.stdout.flush()
            
        if (i + 1) > mcmc_iterBurn:   
            if ((i + 1) % mcmc_thin) == 0:
                j+=1
            
                if nFix > 0:
                    paramFix_store_tmp[j,:] = paramFix
            
                if nRnd > 0:
                    paramRnd_store_tmp[j,:,:] = paramRnd
                    zeta_store_tmp[j,:] = zeta
                    Omega_store_tmp[j,:,:] = Omega
                    Corr_store_tmp[j,:,:], sd_store_tmp[j,:,] = corrcov(Omega)
                    
            if (j + 1) == mcmc_iterMemThin:
                l = ll; ll += mcmc_iterMemThin; sl = slice(l, ll)
                
                print('Storing chain ' + str(chainID + 1))
                sys.stdout.flush()
                
                if nFix > 0:
                    paramFix_store[sl,:] = paramFix_store_tmp
                    
                if nRnd > 0:
                    paramRnd_store[sl,:,:] = paramRnd_store_tmp
                    zeta_store[sl,:] = zeta_store_tmp
                    Omega_store[sl,:,:] = Omega_store_tmp
                    Corr_store[sl,:,:] = Corr_store_tmp
                    sd_store[sl,:,] = sd_store_tmp
                
                j = -1 


# ### Posterior analysis

# In[8]:


def postAna(paramName, nParam, nParam2, mcmc_nChain, mcmc_iterSampleThin, modelName):
    colHeaders = ['mean', 'std. dev.', '2.5%', '97.5%', 'Rhat']
    q = np.array([0.025, 0.975])
    nSplit = 2
    
    postDraws = np.zeros((mcmc_nChain, mcmc_iterSampleThin, nParam, nParam2))
    for c in range(mcmc_nChain):
        file = h5py.File(modelName + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
        postDraws[c,:,:,:] = np.array(file[paramName + '_store']).reshape((mcmc_iterSampleThin, nParam, nParam2))
        
    tabPostAna = np.zeros((nParam * nParam2, len(colHeaders)))
    postMean = np.mean(postDraws, axis = (0,1))
    tabPostAna[:, 0] = np.array(postMean).reshape((nParam * nParam2,))
    tabPostAna[:, 1] = np.array(np.std(postDraws, axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 2] = np.array(np.quantile(postDraws, q[0], axis = (0,1))).reshape((nParam * nParam2,))
    tabPostAna[:, 3] = np.array(np.quantile(postDraws, q[1], axis = (0,1))).reshape((nParam * nParam2,))
    
    m = floor(mcmc_nChain * nSplit)
    n = floor(mcmc_iterSampleThin / nSplit)
    postDrawsSplit = np.zeros((m, n, nParam, nParam2))
    postDrawsSplit[0:mcmc_nChain, :, :, :] = postDraws[:, 0:n, :, :]
    postDrawsSplit[mcmc_nChain:m, :, :, :] = postDraws[:,n:mcmc_iterSampleThin, :, :]
    muChain = np.mean(postDrawsSplit, axis = 1)
    muChainArr = np.array(muChain).reshape((m,1,nParam, nParam2))
    mu = np.array(np.mean(muChain, axis = 0)).reshape((1, nParam, nParam2))
    B = (n / (m - 1)) * np.sum((muChain - mu)**2)
    sSq = (1 / (n - 1)) * np.sum((postDrawsSplit - muChainArr)**2, axis = 1)
    W = np.mean(sSq, axis = 0)
    varPlus = ((n - 1) / n) * W + B / n
    Rhat = np.empty((nParam, nParam2)) * np.nan
    W_idx = W > 0
    Rhat[W_idx] = np.sqrt(varPlus[W_idx] / W[W_idx])
    tabPostAna[:, 4] = np.array(Rhat).reshape((nParam * nParam2,))
    
    if paramName not in ["Omega", "Corr", "paramRnd"]:
        postMean = np.ndarray.flatten(postMean)
        
    pdTabPostAna = pd.DataFrame(tabPostAna, columns = colHeaders) 
    return postMean, pdTabPostAna             


# ### Estimation

# In[9]:


def estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd,
        xFix_trans, xRnd_trans):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRnd_transBool = np.sum(xRnd_trans) > 0  
    
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd = xList[0], xList[1]
    
    ### 
    #Posterior sampling
    ###
    
    mcmc_iter = mcmc_iterBurn + mcmc_iterSample
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterMemThin = floor(mcmc_iterMem / mcmc_thin)

    A = A * np.ones((nRnd,))
    invASq = A ** (-2)
    
    paramFix = paramFix_inits
    zeta = zeta_inits
    Omega = Omega_inits
    
    tic = time.time()

    for c in range(mcmc_nChain):
        mcmcChain(c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,    
                modelName,
                paramFix, zeta, Omega, invASq, nu, diagCov,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    """
    Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChain)(
                c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,    
                modelName,
                paramFix, zeta, Omega, invASq, nu, diagCov,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    for c in range(mcmc_nChain))
    """

    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
        
    ###
    #Posterior analysis
    ###

    if nFix > 0:        
        postMean_paramFix, pdTabPostAna_paramFix = postAna('paramFix', nFix, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Fixed parameters:')    
        print(pdTabPostAna_paramFix)
    else:
        postMean_paramFix = None; pdTabPostAna_paramFix = None;
 
    if nRnd > 0:
        postMean_zeta, pdTabPostAna_zeta = postAna('zeta', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (means):')    
        print(pdTabPostAna_zeta)
        
        postMean_sd, pdTabPostAna_sd = postAna('sd', nRnd, 1, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (standard deviations):')    
        print(pdTabPostAna_sd)
        
        postMean_Omega, pdTabPostAna_Omega = postAna('Omega', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (covariance matrix):')    
        print(pdTabPostAna_Omega)
        
        postMean_Corr, pdTabPostAna_Corr = postAna('Corr', nRnd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
        print(' ')
        print('Random parameters (correlation matrix):')    
        print(pdTabPostAna_Corr)
        
        postMean_paramRnd, pdTabPostAna_paramRnd = postAna('paramRnd', nInd, nRnd, mcmc_nChain, mcmc_iterSampleThin, modelName)
    else:
        postMean_zeta = None; pdTabPostAna_zeta = None;
        postMean_sd = None; pdTabPostAna_sd = None;
        postMean_Omega = None; pdTabPostAna_Omega = None;
        postMean_Corr = None; pdTabPostAna_Corr = None;
        postMean_paramRnd = None; pdTabPostAna_paramRnd = None;
    
    ###
    #Simulate log-likelihood at posterior means
    ###
    
    if nFix > 0 and nRnd == 0:
        simDraws_star = 1
    else:
        simDraws_star = simDraws
    
    pSim = np.zeros((simDraws_star, nInd))
    
    paramFix = 0; paramRnd = 0;
    if nFix > 0: paramFix = postMean_paramFix
    if nRnd > 0: postMean_chOmega = np.linalg.cholesky(postMean_Omega)      
                
    for i in np.arange(simDraws_star):
        if nRnd > 0:
            paramRnd = postMean_zeta + (postMean_chOmega @ np.random.randn(nRnd, nInd)).T
            
        lPInd = probMxl(
                paramFix, paramRnd,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd,
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
        pSim[i, :] = np.exp(lPInd)
    
    logLik = np.sum(np.log(np.mean(pSim, axis = 0)))
    print(' ')
    print('Log-likelihood (simulated at posterior means): ' + str(logLik)) 
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 
        
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc,
               'logLik': logLik,
               'postMean_paramFix': postMean_paramFix, 'pdTabPostAna_paramFix': pdTabPostAna_paramFix,
               'postMean_zeta': postMean_zeta, 'pdTabPostAna_zeta': pdTabPostAna_zeta, 
               'postMean_sd': postMean_sd, 'pdTabPostAna_sd': pdTabPostAna_sd, 
               'postMean_Omega': postMean_Omega, 'pdTabPostAna_Omega': pdTabPostAna_Omega, 
               'postMean_Corr': postMean_Corr, 'pdTabPostAna_Corr': pdTabPostAna_Corr,
               'postMean_paramRnd': postMean_paramRnd, 'pdTabPostAna_paramRnd': pdTabPostAna_paramRnd
               }
    
    return results


# ### Prediction

# In[10]:


def mcmcChainPred(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
        modelName,
        xFix, nFix, 
        sim_xRnd, nRnd, 
        nInd, nObs, nRow,
        sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    paramFix_store = None
    if nFix: paramFix_store = np.array(file['paramFix_store'])
    zeta_store = np.array(file['zeta_store'])
    Omega_store = np.array(file['Omega_store'])
    
    ###
    #Simulate
    ###

    pPred = np.zeros((nRow + nObs,))
    vFix = 0 
    
    for i in np.arange(mcmc_iterSampleThin):
        
        if nFix: 
            paramFix = paramFix_store[i,:]
            vFix = np.tile(xFix @ paramFix, (nSim,));
        
        zeta_tmp = zeta_store[i,:]
        ch_tmp = np.linalg.cholesky(Omega_store[i,:,:])
        
        pPred_iter = np.zeros((nRow + nObs,))
        
        for t in np.arange(nTakes):
            paramRnd = zeta_tmp + (ch_tmp @ np.random.randn(nRnd, nInd * nSim)).T
            paramRndPerRow = np.repeat(paramRnd, sim_rowsPerInd, axis = 0)
            vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            
            pPred_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            pPred_iter += pPred_take
            
        pPred += (pPred_iter / nTakes)
        
        if ((i + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (predictive simulation)')
            sys.stdout.flush()
            
    pPred /= mcmc_iterSampleThin
    return pPred
    
def predict(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xRnd):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    
    xList = [xFix, xRnd]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd = xList[0], xList[1]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1))
    sim_rowsPerInd = np.tile(rowsPerInd, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    
    pPred = np.zeros((nObs + nRow,))
    for c in np.arange(mcmc_nChain):
        predPred_chain = mcmcChainPred(
                c, seed,
                mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
                modelName,
                xFix, nFix, 
                sim_xRnd, nRnd, 
                nInd, nObs, nRow,
                sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx)
        pPred += predPred_chain
    pPred /= mcmc_nChain
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred


# ### Generate fake data

# In[11]:


import sys
RUN = int(sys.argv[1])
#RUN = 1
print("RUN number:", RUN)

np.random.seed(RUN)
    
"""
###
#Load data
###

data = pd.read_csv('swissmetro_long.csv')
data = data[((data['PURPOSE'] != 1) & (data['PURPOSE'] != 3)) != True]
data = data[data['ID'] <= 200]

###
#Prepare data
###

indID = np.array(data['indID'].values, dtype = 'int64')
obsID = np.array(data['obsID'].values, dtype = 'int64')
altID = np.array(data['altID'].values, dtype = 'int64')

chosen = np.array(data['chosen'].values, dtype = 'int64')

tt = np.array(data['TT'].values, dtype = 'float64') / 10
cost = np.array(data['CO'].values, dtype = 'float64') / 10
he = np.array(data['HE'].values, dtype = 'float64')/ 10
ga = np.array(data['GA'].values, dtype = 'int64')
cost[(altID <= 2) & (ga == 1)] = 0

const2 = 1 * (altID == 2)
const3 = 1 * (altID == 3)
"""
###
#Generate data
###

N = 2000
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


# ### Run Gibbs sampler

# In[12]:


###
#Estimate MXL via MCMC
###

#xFix = np.stack((const2, const3), axis = 1)
#xRnd = -np.stack((cost, tt), axis = 1) #np.zeros((0,0)) #-np.hstack((cost, he, tt))

#Fixed parameter distributions
#0: normal
#1: log-normal (to assure that fixed parameter is striclty negative or positive)
xFix_trans = np.array([0, 0, 0, 0])

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRnd_trans = np.array([0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
Omega_inits = 0.1 * np.eye(xRnd.shape[1])

A = 1.04
nu = 2
diagCov = False

mcmc_nChain = 2
mcmc_iterBurn = 20000
mcmc_iterSample = 20000
mcmc_thin = 5
mcmc_iterMem = 20000
mcmc_disp = 1000
seed = RUN
simDraws = 1000    

rho = 0.1
rhoF = 0.01

modelName = 'test'
deleteDraws = False

results = estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        A, nu, diagCov,
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd,
        xFix_trans, xRnd_trans)


# ## Get results for comparison with other methods

# In[13]:


# convert long format to wide format
xs = []
ys = []
num_resp = N
num_alternatives = J
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


# In[14]:


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


# In[15]:


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
        #print(beta_resp.shape)

        for resp_id in range(num_resp):
            # calculate utilities based on params
            utilities = np.vstack([np.dot(X[resp_id,:,np.where(alt_ids == i)[0]].T, 
                                          beta_resp[resp_id, np.where(alt_ids == i)[0]]) for i in range(num_alternatives)])

            # adjust utility for unavailable alternatives
            utilities = utilities.T + alt_av[resp_id]
            #print(utilities.shape)

            # likelihood
            probs = softmax(utilities, axis=1)
            lPInd = np.sum(np.log(probs[np.arange(T), y[resp_id]]))

            pSim[i, resp_id] = np.exp(lPInd)

    logLik = np.sum(np.log(np.mean(pSim, axis=0)))
    
    return logLik


# In[16]:


np.set_printoptions(precision=3)

alpha_params = results["postMean_paramFix"]
beta_params = results["postMean_zeta"]
params_resps = results["postMean_paramRnd"]
Omega_params = results["postMean_Omega"]
            
alpha_error = np.abs(true_alpha - alpha_params).mean()
alpha_rmse = np.sqrt(np.mean((true_alpha - alpha_params)**2))
beta_error = np.abs(true_beta - beta_params).mean()
beta_rmse = np.sqrt(np.mean((true_beta - beta_params)**2))
params_resps_error = np.abs(betaInd_tmp - params_resps).mean()
params_resps_rmse = np.sqrt(np.mean((betaInd_tmp - params_resps)**2))
Omega_rmse = np.sqrt(np.mean((true_Omega - Omega_params)**2))

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

loglik, acc = loglikelihood(alt_attributes, true_choices, np.zeros((N,T,J)), 
                            alpha_params, beta_params, params_resps)
print("\nLoglikelihood:", loglik)

loglik_hyp,_ = loglikelihood(alt_attributes, true_choices, np.zeros((N,T,J)), 
                             alpha_params, beta_params, np.tile(beta_params, [N,T]))
print("\nLoglikelihood (hyper-priors only):", loglik_hyp)

sim_loglik = sim_loglikelihood(alt_attributes, true_choices, np.zeros((N,T,J)), 
                               results["postMean_paramFix"], results["postMean_zeta"], 
                               np.linalg.cholesky(results["postMean_Omega"]))
print("\nLoglikelihood (simulated at posterior means):", sim_loglik)


# In[17]:


import os
BATCH_SIZE = num_resp
output_dir = "Results_FakeData_N%d_T%d_J%d_L%d_K%d_Corr%.1f_Scale%.1f_Batch%d" % (N,T,J,L,K,
                                                                                   corr,scale_factor,
                                                                                   BATCH_SIZE)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fname = output_dir + "/Gibbs.txt"
if not os.path.exists(fname):
    fw = open(fname, "w")
    fw.write("Run\tTime\tLoglik\tSim. Loglik\tLoglik (hyper)\tRMSE alpha\tRMSE beta\tRMSE betaInd\tRMSE Omega\n")
else:
    fw = open(fname, "a")
    
fw.write("%d\t%.0f\t%.1f\t%.1f\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\n" % (RUN, results["estimation_time"], 
                                                            loglik, sim_loglik, loglik_hyp, 
                                                            alpha_rmse, beta_rmse, params_resps_rmse, Omega_rmse))
fw.close()

