# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:39:31 2022

@author: yexue
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import time
from numpy.random import random


# %matplotlib inline
seed = 122
np.random.seed(seed)
#%%
# INITIALIZE
seed = 1401
np.random.seed(seed)

#initializing parameters
T = 100  # Number of time steps.
t = range(T)  # Time.
x0 = 0 #% Initial state. normal(0,1)
true_x = np.zeros(T)  # Hidden states X_n
y = np.zeros(T)  # Observations Y_n for model 1
true_x[0] = x0  # Initial state X0

rho = 0.8
tau = 1
sigma = 0.5


# % Linear–quadratic–Gaussian model parameters

initVar = 1 #initial variance
v = np.random.normal(size=T)
w = np.random.normal(size=T)

# GENERATE TRUE STATE AND MEASUREMENTS:
y[0] = true_x[0] + sigma * w[0]
for t in range(1, T):
    true_x[t] = rho * true_x[t - 1] + tau * v[t]
    y[t] = true_x[t] + sigma * w[t]
    
plt.plot(true_x)
plt.scatter(range(T),y, color="orange")
plt.xlabel("time n")
# plt.ylabel(r"$Y_n$")
plt.title(r"$x^*_{0:T}$")
# plt.legend()

#%% q3.1 Kalman filter
# % Kalman filter to generate recursive likelihood p(y0:T)

def KalmanFilter(rho_1, tau2_1):

    RecLikeMean = np.zeros(T)
    RecLikeVar = np.zeros(T) 
    # % KF
    mu = np.zeros(T)
    Sigma = np.zeros(T)
    mu[0] = x0
    Sigma[0] = initVar

    a = rho_1
    b = np.sqrt(tau2_1)
    c = 1
    d = 0.5

    RecLikeMean[0] = x0
    RecLikeVar[0] = c * initVar * c + d * d

    for t in range(1, T):
        mu_pred = a * mu[t - 1]

        SigmaPred = a * Sigma[t - 1] * a + b * b
    #     print(SigmaPred)

        z = y[t] - c * mu_pred
        SS = c * SigmaPred * c + d * d

        RecLikeMean[t] = mu_pred
        RecLikeVar[t] = SS

        K = SigmaPred * c / SS
        mu[t] = mu_pred + K * z
        Sigma[t] = (1 - K * c) * SigmaPred

    # plt.plot(true_x)
    # plt.plot(RecLikeMean, "--")
    rec_true = stats.norm.pdf(y, RecLikeMean, np.sqrt(RecLikeVar))
    return(rec_true)
    # print(rec_true)
    # plt.plot(np.cumsum(np.log(rec_true)))
    # plt.title(r"$p_\theta(y_{0:T})$ true log-likelihood from Kal")
    
#%% q3.1 ideal MCMC

# def inverse_gamma_unnormalised(x, a, b):
#     return (x ** (-a - 1)) * np.exp(-(b / x))

def log_joint_prior(r, t2):
    return(stats.uniform.logpdf(r, loc=-1, scale=2) + stats.invgamma.logpdf(t2, 1))

rho0 = 0.2
tau2_0 = 0.1

rep = 20000
rw_step_rho = 0.2
rw_step_tau2 = 0.1

RATIO_KF = []
RHO_KF = np.zeros(rep)
TAU2_KF = np.zeros(rep)

logPrior = log_joint_prior(rho0, tau2_0)
logYsum = sum(np.log(KalmanFilter(rho0, tau2_0))) + logPrior

rhoSample = rho0
tau2Sample = tau2_0


tic = time.time()                    
for i in range(rep):
    if i%2000 == 0:
        toc = time.time()
        print(f"iteration {i}, total run time: {toc-tic}")
    rhoNew = rhoSample + rw_step_rho * np.random.normal(0,1)
    tau2New = tau2Sample + rw_step_tau2 * np.random.normal(0,1)
    while(rhoNew < -1 or rhoNew > 1):
        rhoNew = rhoSample + rw_step_rho * np.random.normal(0,1)
    while(tau2New <= 0):
        tau2New = tau2Sample + rw_step_tau2 * np.random.normal(0,1)
    
    logPriorNew = log_joint_prior(rhoNew, tau2New)
    logYsumNew = sum(np.log(KalmanFilter(rhoNew, tau2New)))+ logPriorNew
    
    ratio = np.exp(logYsumNew - logYsum)
    accept_ratio=min(1,ratio)
    RATIO_KF.append(accept_ratio)
    
    u = np.random.uniform(0,1)
    if (u <= accept_ratio):
        logYsum=logYsumNew
        rhoSample = rhoNew
        tau2Sample = tau2New
    RHO_KF[i] = rhoSample
    TAU2_KF[i] = tau2Sample

                    
print(f"Done! Total run time: {toc-tic}")
# RATIO_KF = RATIO
# RHO_KF = RHO
# TAU2_KF = TAU2

#%% q3.1 plots

fig2, axs = plt.subplots(1, 2, figsize=(9,4))
sns.distplot(ax=axs[0], x=RHO_KF, hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
axs[0].set_title(r"$\rho$")
axs[0].axvline(x=np.mean(RHO_KF), 
   label =fr"$E[\rho] = {np.round(np.mean(RHO_KF), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[0].legend()

sns.distplot(ax=axs[1], x=TAU2_KF, hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
axs[1].set_title(r"$\tau^2$")
axs[1].axvline(x=np.mean(TAU2_KF), 
   label =fr"$E[\tau^2] = {np.round(np.mean(TAU2_KF), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[1].legend()

plt.savefig("plots/q3_1.pdf")

#%% q3.1 diagnostics
from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(TAU2_KF, lags=50)

print(np.mean(RATIO_KF))
plt.figure(figsize=(6, 2), dpi=80)
plt.plot(np.cumsum(RATIO_KF)/range(len(RATIO_KF)))
plt.title(f"Average acceptance Ratio {round(np.mean(RATIO_KF),4)}")
plt.xlabel("Iteration n")
plt.ylabel(r"$\alpha$")
plt.savefig("plots/q3_1_RATIO.pdf")
#%%
fig, axs = plt.subplots(2, 1, figsize=(15,4))
axs[0].plot(RHO_KF, label=r"$\rho$")
axs[0].axhline(y=0.8, color="orange", linestyle="--", label=r"true $\rho$")
axs[0].legend()
axs[1].plot(TAU2_KF, label=r"$\tau^2$")
axs[1].axhline(y=1, color="orange", linestyle="--", label=r"true $\tau^2$")
axs[1].legend(loc="upper right")
axs[0].set_title(r"Ideal MMH trace plots for $\rho$ and $\tau^2$")
plt.savefig("plots/q3_1_trace.pdf")




#%% q3.2 a) SIR
def un_log_norm_pdf(y, x):
    return(-0.5*np.log(2*np.pi)-np.log(0.5)-(y-x)**2/(0.5**2)/2)

def SMC_SIR(r, t2):
    N = 150 # particle size
    a = r
    b = np.sqrt(t2)
    c = 1
    d = sigma
    
    x = np.zeros((N,T))
    xu=np.zeros((N,T));
    q = np.zeros((N,T))
    qq = np.zeros((N,T))
    log_w=np.zeros((N,T))
    I = np.zeros((N,T))
    log_rec=np.zeros(T)

    xu[:,0] = np.random.normal(0,1,size=N)
    log_w[:, 0] = list(map(lambda x_u: stats.norm.logpdf(y[0], x_u, d), xu[:,0]))   
    qq[:, 0] = np.exp(log_w[:, 0])
    q[:, 0] = qq[:, 0] / sum(qq[:, 0])
    I = multinomial_resample(q[:, 0])
    I = I.astype(int)
    x[:, 0] = xu[I, 0]
    log_rec[0] = np.log(np.mean(qq[:,0]))
    
    for t in range(T-1):
        xu[:, t+1] = a*x[:,t] + b * np.random.normal(size=N)
        # compute weights and normalise
#         tic1 = time.time()
#         log_w[:, t+1] = list(map(lambda x_u: stats.norm.logpdf(y[t+1], x_u, d), xu[:,t+1]))
        log_w[:, t+1] = list(map(lambda x_u: un_log_norm_pdf(y[t+1], x_u), xu[:,t+1]))
        offset = max(log_w[:,t+1])
        log_w[:,t+1] = log_w[:,t+1] - offset
#         toc1 = time.time()
#         print(toc1-tic1)
#         tic1 = time.time()
        qq[:,t+1]=np.exp(log_w[:,t+1])
        q[:,t+1] = qq[:,t+1]/sum(qq[:,t+1])

        log_rec[t+1]=np.log(np.mean(qq[:,t+1]))+offset

        It = multinomial_resample(q[:, t+1])
        It = It.astype(int)

        x[:, t + 1] = xu[It, t + 1].copy()  
        x[:, :t] = x[It, :t].copy()
        
    return(log_rec)

#%% q3.2 a)


# seed = 18411
# np.random.seed(seed)

rho0 = 0.2
tau2_0 = 0.1

# rep = 20000
rep = 5000
rw_step_rho = 0.2
rw_step_tau2 = 0.1


RATIO = []
RHO = np.zeros(20000)
TAU2 = np.zeros(20000)

# SMC: SIR
logPrior =  log_joint_prior(rho0, tau2_0)
logYsum = sum((SMC_SIR(rho0, tau2_0))) + logPrior

rhoSample = rho0
tau2Sample = tau2_0

print("init")
#%%
# rep = 1000
# 2nd: 1000-5000
# 3rd: 5000-10000

# rw_step_tau2 = 0.8

tic = time.time()          
for i in range(10000, 20000):
#     print(i)
    if i%100 == 0:
        toc = time.time()
        print(f"iteration {i}, total run time: {toc-tic}")
    rhoNew = rhoSample + rw_step_rho * np.random.normal(0,1)
    tau2New = tau2Sample + rw_step_tau2 * np.random.normal(0,1)
#     while(rhoNew < -1 or rhoNew > 1):
#         rhoNew = rhoSample + rw_step * np.random.normal(0,1)
    while(tau2New <= 0):
        tau2New = tau2Sample + rw_step_tau2 * np.random.normal(0,1)
    
    logPriorNew = log_joint_prior(rhoNew, tau2New)
    logYsumNew = sum(SMC_SIR(rhoNew, tau2New)) + logPriorNew
    
    ratio = np.exp(logYsumNew - logYsum)
    accept_ratio=min(1,ratio)
    RATIO.append(accept_ratio)
    
    u = np.random.uniform(0,1)
    if (u <= accept_ratio):
        logYsum=logYsumNew
        rhoSample = rhoNew
        tau2Sample = tau2New
    RHO[i] = rhoSample
    TAU2[i] = tau2Sample

toc = time.time()                
print(f"Done! Total run time: {toc-tic}")

#%% q3.2 a) plot
rep2 = 5000
# RHO_PF = RHO
# TAU2_PF = TAU2

fig, axs = plt.subplots(1, 2, figsize=(9,4))
sns.distplot(ax=axs[0], x=RHO_PF[:rep2], hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
sns.distplot(ax=axs[0], x=RHO_KF, hist = False, kde = True,
                 kde_kws = {'linewidth': 1, "label":"KF density"})
axs[0].set_title(fr"$\rho$")
axs[0].axvline(x=np.mean(RHO_PF[:rep2]), 
   label =fr"$E[\rho] = {np.round(np.mean(RHO[:rep2]), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[0].legend()

sns.distplot(ax=axs[1], x=TAU2_PF[:rep2], hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
sns.distplot(ax=axs[1], x=TAU2_KF, hist = False, kde = True,
                 kde_kws = {'linewidth': 1, "label":"KF density"})
axs[1].set_title(fr"$\tau^2$")
axs[1].axvline(x=np.mean(TAU2_PF[:rep2]), 
   label =fr"$E[\tau^2] = {np.round(np.mean(TAU2[:rep2]), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[1].legend()

# plt.savefig("q3_2_a_20000.pdf")
#%%
rep2 = 10000
fig, axs = plt.subplots(1, 2, figsize=(9,4))
sns.distplot(ax=axs[0], x=RHO[:rep2], hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
sns.distplot(ax=axs[0], x=RHO_KF, hist = False, kde = True,
                 kde_kws = {'linewidth': 1, "label":"KF density"})
axs[0].set_title(fr"$\rho$")
axs[0].axvline(x=np.mean(RHO[:rep2]), 
   label =fr"$E[\rho] = {np.round(np.mean(RHO[:rep2]), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[0].legend()

sns.distplot(ax=axs[1], x=TAU2[:rep2], hist = True, kde = True,
                 kde_kws = {'linewidth': 1, "label":"density"})
sns.distplot(ax=axs[1], x=TAU2_KF, hist = False, kde = True,
                 kde_kws = {'linewidth': 1, "label":"KF density"})
axs[1].set_title(fr"$\tau^2$")
axs[1].axvline(x=np.mean(TAU2[:rep2]), 
   label =fr"$E[\tau^2] = {np.round(np.mean(TAU2[:rep2]), 4)}$",
   color = "red", 
   alpha = 0.7,
   linestyle = "--")
axs[1].legend()
# print(np.mean(RATIO))
plt.savefig("plots/q3_2_a.pdf")


#%%
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(TAU2, lags=50)

# print(np.mean(RATIO))
plt.figure(figsize=(6, 2), dpi=80)
plt.plot(np.cumsum(RATIO)/range(len(RATIO)))
plt.title(f"Average acceptance Ratio {round(np.mean(RATIO),4)}")
plt.xlabel("Iteration n")
plt.ylabel(r"$\alpha$")
plt.savefig("plots/q3_2_a_RATIO.pdf")


#%%
fig, axs = plt.subplots(2, 1, figsize=(15,4))
axs[0].plot(RHO, label=r"$\rho$")
axs[0].axhline(y=0.8, color="orange", linestyle="--", label=r"true $\rho$")
axs[0].legend()
axs[1].plot(TAU2, label=r"$\tau^2$")
axs[1].axhline(y=1, color="orange", linestyle="--", label=r"true $\tau^2$")
axs[1].legend(loc="upper right")
axs[0].set_title(r"Ideal MMH trace plots for $\rho$ and $\tau^2$")
plt.savefig("plots/q3_2_a_trace.pdf")

# fig, axs = plt.subplots(2, 1, figsize=(15,4))
# axs[0].plot(RHO_KF)
# axs[1].plot(TAU2_KF)
# RHO[2000:6000]

#%% q3.2 d)
rho_pred = np.mean(RHO_PF[:rep2])
tau2_pred = np.mean(TAU2_PF[:rep2])
print(rho_pred)
print(tau2_pred)

#%% SIR to predict X_0:T

N = 150 # particle size
a = rho_pred
b = np.sqrt(tau2_pred*2)
c = 1
d = sigma

x = np.zeros((N,T))
xu=np.zeros((N,T))
q = np.zeros((N,T))
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))
log_rec=np.zeros(T)

xu[:,0] = np.random.normal(0,1,size=N)
log_w[:, 0] = list(map(lambda x_u: stats.norm.logpdf(y[0], x_u, d), xu[:,0]))   
qq[:, 0] = np.exp(log_w[:, 0])
q[:, 0] = qq[:, 0] / sum(qq[:, 0])
I = multinomial_resample(q[:, 0])
I = I.astype(int)
x[:, 0] = xu[I, 0]
# log_rec[0] = np.log(np.mean(qq[:,0]))

for t in range(T-1):
    xu[:, t+1] = a*x[:,t] + b * np.random.normal(size=N)
    # compute weights and normalise
#         tic1 = time.time()
#         log_w[:, t+1] = list(map(lambda x_u: stats.norm.logpdf(y[t+1], x_u, d), xu[:,t+1]))
    log_w[:, t+1] = list(map(lambda x_u: un_log_norm_pdf(y[t+1], x_u), xu[:,t+1]))
    offset = max(log_w[:,t+1])
    log_w[:,t+1] = log_w[:,t+1] - offset
#         toc1 = time.time()
#         print(toc1-tic1)
#         tic1 = time.time()
    qq[:,t+1]=np.exp(log_w[:,t+1])
    q[:,t+1] = qq[:,t+1]/sum(qq[:,t+1])

    log_rec[t+1]=np.log(np.mean(qq[:,t+1]))+offset

    It = multinomial_resample(q[:, t+1])
    It = It.astype(int)

    x[:, t + 1] = xu[It, t + 1].copy()  
    x[:, :t] = x[It, :t].copy()
    
    #%%
    #%% q3.2 d) plots
plt.plot(true_x, "--", color="black", label=r"$x^*_{0:T}$")
plt.plot(sum(q*xu), label=r"$E[X_{0:T}|Y_{0:T}]$")

ci = 1.96 * np.sqrt(sum(q*(xu*xu))-(sum(q*xu)**2))

# plt.plot(RecLikeMean, "--")
plt.fill_between(range(T), (sum(q*xu)-ci), (sum(q*xu)+ci), 
                 color='b', alpha=.1,
                 label="95% CI")
plt.legend()
plt.title(r"Estimation of $X_{0:T}$ from PF against time")
plt.xlabel("time n")
plt.savefig("plots/q3_2_d.pdf")


# sum(q*(xu*xu))-(sum(q*xu)**2)

# plt.plot(np.median(rhos,axis=0))
# plt.plot(np.quantile(rhos, 0.975, axis=0))
# plt.plot(np.quantile(rhos, 0.025, axis=0))
# plt.fill_between(range(T), np.quantile(q*xu*N, 0.025, axis=0), 
#                   np.quantile((q*xu)*N, 0.975, axis=0), color='b', alpha=.1)


#%% KF

RecLikeMean = np.zeros(T)
RecLikeVar = np.zeros(T) 
# % KF
mu = np.zeros(T)
Sigma = np.zeros(T)
mu[0] = x0
Sigma[0] = initVar

a = rho_pred
b = np.sqrt(tau2_pred*2)
c = 1
d = sigma

RecLikeMean[0] = x0
RecLikeVar[0] = c * initVar * c + d * d

for t in range(1, T):
    mu_pred = a * mu[t - 1]

    SigmaPred = a * Sigma[t - 1] * a + b * b
#     print(SigmaPred)

    z = y[t] - c * mu_pred
    SS = c * SigmaPred * c + d * d

    RecLikeMean[t] = mu_pred
    RecLikeVar[t] = SS

    K = SigmaPred * c / SS
    mu[t] = mu_pred + K * z
    Sigma[t] = (1 - K * c) * SigmaPred

# plt.plot(true_x)
# ci = 1.96 * np.sqrt(RecLikeVar)
ci = 1.96 * np.std(RecLikeMean)
plt.plot(true_x, "--", color="black", label=r"$x^*_{0:T}$")
plt.plot(RecLikeMean)
plt.fill_between(range(T), (RecLikeMean-ci), (RecLikeMean+ci), color='b', alpha=.1)



#%% q3.2 d) joint plot
fig, axs = plt.subplots(1,2, figsize=(12,4))
axs[0].plot(true_x, "--", color="black", label=r"$x^*_{0:T}$")
axs[0].plot(sum(q*xu), label=r"$E[X_{0:T}|Y_{0:T}]$")

ci = 1.96 * np.sqrt(sum(q*(xu*xu))-(sum(q*xu)**2))

# plt.plot(RecLikeMean, "--")
axs[0].fill_between(range(T), (sum(q*xu)-ci), (sum(q*xu)+ci), 
                 color='b', alpha=.1,
                 label="95% CI")
# axs[0].title(r"Estimation of $X_{0:T}$ from PF against time")
axs[0].set_title("Particle Filter")
axs[0].set_xlabel("time n")
axs[0].legend()

ci = 1.96 * np.std(RecLikeMean)
axs[1].plot(true_x, "--", color="black", label=r"$x^*_{0:T}$")
axs[1].plot(RecLikeMean, label=r"$E[X_{0:T}|Y_{0:T}]$")
axs[1].fill_between(range(T), (RecLikeMean-ci), (RecLikeMean+ci), 
                    color='b', alpha=.1, label="95% CI")
axs[1].set_title("Kalman Filter")
axs[1].set_xlabel("time n")
axs[1].legend()
plt.savefig("plots/q3_2_d_together.pdf")

#%% q3.4 a)

# N = 100 # particle size
# a = rho
# b = np.sqrt(tau2)
# c = 1
# d = sigma
# SEED = 12347
# np.random.seed(SEED)
N = 500
eps_rho = 0.01
eps_tau2 = 0.01

x = np.zeros((N,T))
xu=np.zeros((N,T))
rhos = np.zeros((N,T))
tau2s = np.zeros((N,T))
rhosu = np.zeros((N,T))
tau2su = np.zeros((N,T))

q = np.zeros((N,T))
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))
# log_rec=np.zeros(T)

xu[:,0] = np.sqrt(initVar) * np.random.normal(0,1,size=N)
rhosu[:,0] = np.random.uniform(low=-1, high=1, size=N)
tau2su[:,0] = 1. / np.random.gamma(1, 1, size=N)

log_w[:, 0] = list(map(lambda x_u: stats.norm.logpdf(y[0], x_u, sigma), xu[:,0]))   
qq[:, 0] = np.exp(log_w[:, 0])
q[:, 0] = qq[:, 0] / sum(qq[:, 0])
# I[:, 0] = np.random.choice(range(N), N, p=q[:, 0])
# I[:, 0] = np.searchsorted(np.cumsum(q[:, 0]), random(N))
# I[:, 0] = stats.multinomial(x=range(N), p=q[:, 0])
# I[:, 0] = np.random.multinomial(N, q[:, 0], N)
I = multinomial_resample(q[:, 0])

I = I.astype(int)
x[:, 0] = xu[I, 0]
rhos[:, 0] = rhosu[I, 0]
tau2s[:, 0] = tau2su[I, 0]


# log_rec[0] = np.log(np.mean(qq[:,0]))

for t in range(T-1):
    for i in range(N):
        xu[i, t+1] = rhos[i, t]*x[i, t] + np.sqrt(tau2s[i, t])*np.random.normal(size=1)
#     xu[:, t+1] = a*x[:,t] + b * np.random.normal(size=N)
#     epsilon = eps * np.random.normal(0,1,size=N)
    rhosu[:, t+1] = rhos[:, t] + eps_rho * np.random.normal(size=N)
    tau2su[:,t+1] = np.abs(tau2s[:, t] + eps_tau2 * np.random.normal(size=N))
    
    # compute weights and normalise
    log_w[:, t+1] = list(map(lambda x_u: un_log_norm_pdf(y[t+1], x_u), xu[:,t+1]))
    offset=max(log_w[:,t+1])
    log_w[:,t+1]=log_w[:,t+1]-offset
    qq[:,t+1]=np.exp(log_w[:,t+1])
    q[:,t+1] = qq[:,t+1]/sum(qq[:,t+1])
    
    It = multinomial_resample(q[:, t+1])
    It = It.astype(int)

    x[:, t + 1] = xu[It, t + 1].copy()  
    x[:, :t] = x[It, :t].copy()
    
    rhos[:, t+1] = rhosu[It, t + 1].copy()
    tau2s[:, t+1] = tau2su[It, t + 1].copy()
    rhos[:, :t] = rhos[It, :t].copy()
    tau2s[:, :t] = tau2s[It, :t].copy()
    

plt.plot(np.median(rhosu, axis=0), label="rho")
plt.plot(np.median(tau2su, axis=0), label="tau")
plt.axhline(y=0.8, linestyle="--", color="blue",
               label=r"true $\rho=0.8$")
plt.axhline(y=1, linestyle="--", color="orange",
               label=r"true $\tau^2=1$")
plt.xlabel("time n")
plt.legend()

#%%
N=200
eps_rho = 0.1
eps_tau2 = 0.1

x = np.zeros((N,T))
xu=np.zeros((N,T))
rhos = np.zeros((N,T))
tau2s = np.zeros((N,T))
rhosu = np.zeros((N,T))
tau2su = np.zeros((N,T))

q = np.zeros((N,T))
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))
# log_rec=np.zeros(T)

xu[:,0] = np.sqrt(initVar) * np.random.normal(0,1,size=N)
rhosu[:,0] = np.random.uniform(low=-1, high=1, size=N)
tau2su[:,0] = 1. / np.random.gamma(1, 1, size=N)
log_w[:, 0] = list(map(lambda x_u: stats.norm.logpdf(y[0], x_u, sigma), xu[:,0]))   
qq[:, 0] = np.exp(log_w[:, 0])
q[:, 0] = qq[:, 0] / sum(qq[:, 0])
I = multinomial_resample(q[:, 0])
I = I.astype(int)
x[:, 0] = xu[I, 0]
rhos[:, 0] = rhosu[I, 0]
tau2s[:, 0] = tau2su[I, 0]

for t in range(T-1):
    for i in range(N):
        xu[i, t+1] = rhos[i, t]*x[i, t] + np.sqrt(tau2s[i, t])*np.random.normal(size=1)

    rhosu[:, t+1] = rhos[:, t] + eps_rho * np.random.normal(size=N)
    tau2su[:,t+1] = np.abs(tau2s[:, t] + eps_tau2 * np.random.normal(size=N))
    
    # compute weights and normalise
    log_w[:, t+1] = list(map(lambda x_u: un_log_norm_pdf(y[t+1], x_u), xu[:,t+1]))
    offset=max(log_w[:,t+1])
    log_w[:,t+1]=log_w[:,t+1]-offset
    qq[:,t+1]=np.exp(log_w[:,t+1])
    q[:,t+1] = qq[:,t+1]/sum(qq[:,t+1])
    
    It = multinomial_resample(q[:, t+1])
    It = It.astype(int)

    x[:, t + 1] = xu[It, t + 1].copy()  
    x[:, :t] = x[It, :t].copy()
    
    rhos[:, t+1] = rhosu[It, t + 1].copy()
    tau2s[:, t+1] = tau2su[It, t + 1].copy()
    rhos[:, :t] = rhos[It, :t].copy()
    tau2s[:, :t] = tau2s[It, :t].copy()
    # rhos[:, t] = rhos[It, t+1]
    # tau2s[:,t+1] = tau2s[It, t+1]

plt.plot(np.median(rhos, axis=0), label="rho")
plt.plot(np.median(tau2s, axis=0), label="tau")
plt.axhline(y=0.8, linestyle="--", color="blue",
               label=r"true $\rho=0.8$")
plt.axhline(y=1, linestyle="--", color="orange",
               label=r"true $\tau^2=1$")
plt.xlabel("time n")
plt.legend()



#%% q3.4.a
fig1, axs = plt.subplots(1, 2, figsize=(10,4))

axs[0].plot(np.median(rhosu,axis=0),
            label=r"median($\rho|y_{0:n}$)")
# plt.plot(np.quantile(rhos, 0.975, axis=0))
# plt.plot(np.quantile(rhos, 0.025, axis=0))
axs[0].fill_between(range(T), np.quantile(rhosu, 0.025, axis=0), 
                 np.quantile(rhosu, 0.975, axis=0), color='b', alpha=.1,
                 label="95% CI")
axs[0].axhline(y=0.8, linestyle="--", color="black",
               label=r"true $\rho=0.8$")
axs[0].set_xlabel("time n")
# axs[0].set_ylim((0,1))
axs[0].legend()

axs[1].plot(np.median(tau2su,axis=0),
            label=r"median($\tau^2|y_{0:n}$)")
axs[1].fill_between(range(T), np.quantile(tau2su, 0.025, axis=0), 
                 np.quantile(tau2su, 0.975, axis=0), color='b', alpha=.1,
                 label="95% CI")
axs[1].axhline(y=1, linestyle="--", color="black",
               label=r"true $\tau^2=1$")
axs[1].set_xlabel("time n")
axs[1].legend()
axs[1].set_ylim((0,5))

plt.savefig("plots/q3_4_a_tmp4.pdf")
# print(np.median(tau2s,axis=0))

# plt.plot(np.median(tau2s[:, 20:],axis=0))
# plt.fill_between(range(20, T), np.quantile(tau2s[:, 20:], 0.025, axis=0), 
#                  np.quantile(tau2s[:, 20:], 0.975, axis=0), color='b', alpha=.1)
#%% q3.4.b
plt.plot(true_x, "--", color="black", label=r"$x^*_{0:T}$")
plt.plot(sum(q*xu), label=r"$E[X_{0:T}|Y_{0:T}]$")

ci = 1.96 * np.sqrt(sum(q*(xu*xu))-(sum(q*xu)**2))

# plt.plot(RecLikeMean, "--")
plt.fill_between(range(T), (sum(q*xu)-ci), (sum(q*xu)+ci), 
                 color='b', alpha=.1,
                 label="95% CI")
plt.legend()
plt.title(r"Estimation of $X_{0:T}$ from extended state")
plt.xlabel("time n")
plt.savefig("plots/q3_4_b_tmp4.pdf")