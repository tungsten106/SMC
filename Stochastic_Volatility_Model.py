# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:55:35 2022

@author: yexue
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import time
from numpy.random import random

# %matplotlib inline
import warnings; warnings.simplefilter('ignore')  # hide warnings
#%%
# INITIALIZE
seed = 135
# seed = 1350
np.random.seed(seed)

#initializing parameters
T = 100  # Number of time steps.
t = range(T)  # Time.
x0 = 0 #% Initial state. normal(0,1)
true_x = np.zeros(T)  # Hidden states X_n
y = np.zeros(T)  # Observations Y_n for model 1
true_x[0] = x0  # Initial state X0

rho = 0.91
sigma = 1
beta = 0.5


# % Linear–quadratic–Gaussian model parameters
a = rho # rho
# b = 1 # tau
b = sigma
c = 0
d = beta
initVar = 1 #initial variance
v = np.random.normal(size=T)
w = np.random.normal(size=T)

# GENERATE TRUE STATE AND MEASUREMENTS:
y[0] = c * true_x[0] + d * v[0]
for t in range(1, T):
    true_x[t] = a * true_x[t - 1] + b * v[t]
    y[t] = d * np.exp(true_x[t]/2) * w[t]
    
plt.plot(true_x)
plt.scatter(range(T),y, color="orange")
plt.xlabel("time n")
# plt.ylabel(r"$Y_n$")
# plt.legend()

#%% resampling function
def multinomial_resample(weights):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.
   Parameters
   ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    return np.searchsorted(cumulative_sum, random(len(weights)))

def residual_resample(weights):
    """ Performs the residual resampling algorithm used by particle filters.
    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    References
    ----------
    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes

#%% N=500
N=500

# filtering approx
x = np.zeros((N,T))
# % resampled particles
xu=np.zeros((N,T))
#normalized weights
q = np.zeros((N,T))
# unormalized weights
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))

# STORE the filtering
xt = np.zeros((N,T))

log_rec=np.zeros(T)

# % INIT: SAMPLE FROM THE PRIOR:
xu[:,0] = np.random.normal(0,1,size=N)

log_w[:, 0] = list(map(lambda x_u: stats.norm.pdf(y[0], 0, beta * np.exp(x_u)), xu[:,0])) 

qq[:, 0] = log_w[:, 0]
q[:, 0] = qq[:, 0] / sum(qq[:, 0])

# log_rec[0] = np.log(np.mean(qq[:, 0])) + offset

# resample
I[:, 0] = multinomial_resample(q[:, 0])
I = I.astype(int)
x[:, 0] = xu[I[:, 0], 0].copy()
xt[:, 0] = x[:, 0].copy()

for t in range(T-1):
    w = b * np.random.normal(size=N)
    xu[:, t+1] = a*x[:,t]+w
    
    d_t = beta * np.exp(x[:,t]/2)

    # compute weights and normalise
    qq[:,t+1] = list(map(lambda d: stats.norm.pdf(y[t+1], 0, d), d_t))
    q[:, t + 1] = qq[:, t + 1] / sum(qq[:, t + 1])
    I[:, t+1] = multinomial_resample(q[:, t+1])


    I = I.astype(int)

    #     % resampling
    x[:, t + 1] = xu[I[:, t+1], t + 1].copy()  
    xt[:, t+1] = x[:,t+1].copy()
    x[:, :t] = x[I[:, t+1], :t].copy()
    
#%% FFBSa
nn = N
idx = np.zeros((nn, T), dtype=int)
idx[:, -1] = multinomial_resample(q[:, -1])


for t in reversed(range(T-1)):
    w_f = np.zeros(nn)
    for i in range(nn):
        x_t = xu[idx[:,t], t]
        w_f[i] = q[i, t]*stats.norm.pdf(xu[i, t+1], rho*x_t[i], sigma)
    w_f = w_f/sum(w_f)
    idx[:, t] = multinomial_resample(w_f)
paths = [xu[idx[:, t], t] for t in range(T)]
paths = np.array(paths)

#%% plot N=500
fig1, axs = plt.subplots(3, 1,figsize=(8,10))
for i, n in enumerate([10, 50, 90]):
    print(i, n)
    # axs[i].figure(figsize=(8, 4))
    sns.distplot(ax=axs[i], x=xt[:,n], hist = False, 
                 # hist_kws={'weights': q[:,n]},
                 kde = True,
                     kde_kws = {'weights': q[:,n],'linewidth': 1, 
                                'label': "(a) filter density"})
    sns.distplot(ax=axs[i], x=x[:,n], hist = False, 
                 kde = True,
                     kde_kws = {'linewidth': 1, 'label': "(a) smoothing density"})
    # plt.hist(x[:,n], density=True)
    sns.distplot(ax=axs[i], x=paths[n], hist = False, 
                 kde = True,
                     kde_kws = {'linewidth': 1, 'label': "(b) FFBSa density"})
    axs[i].axvline(x=true_x[n], 
       label ="true x",
       color = "black", 
       alpha = 0.7,
       linestyle = "--")
    axs[i].set_title(f"N={N}, n={n}")
    axs[i].legend()
plt.savefig(f"plots/q2_1_2_N{N}.pdf")





#%% N=50
N=50

# filtering approx
x = np.zeros((N,T))
# % resampled particles
xu=np.zeros((N,T))
#normalized weights
q = np.zeros((N,T))
# unormalized weights
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))

# STORE the filtering
xt = np.zeros((N,T))

log_rec=np.zeros(T)

# % INIT: SAMPLE FROM THE PRIOR:
xu[:,0] = np.random.normal(0,1,size=N)

log_w[:, 0] = list(map(lambda x_u: stats.norm.pdf(y[0], 0, beta * np.exp(x_u)), xu[:,0])) 

qq[:, 0] = log_w[:, 0]
q[:, 0] = qq[:, 0] / sum(qq[:, 0])

# log_rec[0] = np.log(np.mean(qq[:, 0])) + offset

# resample
I[:, 0] = multinomial_resample(q[:, 0])
I = I.astype(int)
x[:, 0] = xu[I[:, 0], 0].copy()
xt[:, 0] = x[:, 0].copy()

for t in range(T-1):
    w = b * np.random.normal(size=N)
    xu[:, t+1] = a*x[:,t]+w
    
    d_t = beta * np.exp(x[:,t]/2)

    # compute weights and normalise
    qq[:,t+1] = list(map(lambda d: stats.norm.pdf(y[t+1], 0, d), d_t))
    q[:, t + 1] = qq[:, t + 1] / sum(qq[:, t + 1])
    I[:, t+1] = multinomial_resample(q[:, t+1])


    I = I.astype(int)

    #     % resampling
    x[:, t + 1] = xu[I[:, t+1], t + 1].copy()  
    xt[:, t+1] = x[:,t+1].copy()
    x[:, :t] = x[I[:, t+1], :t].copy()
    
nn = N
idx = np.zeros((nn, T), dtype=int)
idx[:, -1] = multinomial_resample(q[:, -1])


for t in reversed(range(T-1)):
    w_f = np.zeros(nn)
    for i in range(nn):
        x_t = xu[idx[:,t], t]
        w_f[i] = q[i, t]*stats.norm.pdf(xu[i, t+1], rho*x_t[i], sigma)
    w_f = w_f/sum(w_f)
    idx[:, t] = multinomial_resample(w_f)
paths = [xu[idx[:, t], t] for t in range(T)]
paths = np.array(paths)

#%% plot N=50

fig1, axs = plt.subplots(3, 1,figsize=(8,10))

for n in [90]:
    # plt.figure(figsize=(8, 4))
    sns.distplot(ax=axs[2], x=xt[:,n], hist = False, 
                 # hist_kws={'weights': q[:,n]},
                 kde = True,
                     kde_kws = {'weights': q[:,n],'linewidth': 1, 
                                'label': "(a) filter density"})
    sns.distplot(ax=axs[2], x=x[:,n], hist = False, 
                 kde = True,
                     kde_kws = {'linewidth': 1, 'label': "(a) smoothing density"})
    # plt.hist(x[:,n], density=True)
    sns.distplot(ax=axs[2], x=paths[n], hist = False, 
                 kde = True,
                     kde_kws = {'linewidth': 1, 'label': "(b) FFBSa density"})
    axs[2].axvline(x=true_x[n], 
       label ="true x",
       color = "black", 
       alpha = 0.7,
       linestyle = "--")
    axs[2].set_title(f"N={N}, n={n}")
    axs[2].legend()
    # plt.savefig(f"plots/q2_1_2_n{n}_N{N}.pdf")
    
for i, n in enumerate([10,50]):
    # plt.figure(figsize=(8, 4))
    sns.distplot(ax=axs[i], x=xu[:,n], hist = True, kde = False,
                  hist_kws = {'label': "(a) filter density"},
                      kde_kws = {'linewidth': 1, 'label': "(a) filter density"})
    sns.distplot(ax=axs[i], x=x[:,n], hist = True,kde = False,
                  hist_kws = {'label': "(a) smoothing density"},
                      kde_kws = {'linewidth': 1, 'label': "(a) smoothing density"})
    # sns.histplot(x[:,n], stat="density", color="green", alpha=0.1)
    #     plt.hist(x[:,n], density=True)
    sns.distplot(ax=axs[i], x=paths[n], hist = True, kde = False,
                  hist_kws = {'label': "(b) FFBSa density"},
                      kde_kws = {'linewidth': 1, 'label': "(b) FFBSa density"})
    axs[i].axvline(x=true_x[n], 
        label ="true x",
        color = "black", 
        alpha = 0.7,
        linestyle = "--")
    axs[i].set_title(f"N={N}, n={n}")
    axs[i].legend()
plt.savefig(f"plots/q2_1_2__N{N}.pdf")




#%% q2.1.3 SIR N=100
N=100
# seed = 510
seed = 1350
np.random.seed(seed)

# filtering approx
x = np.zeros((N,T))
# % resampled particles
xu=np.zeros((N,T))
#normalized weights
q = np.zeros((N,T))
# unormalized weights
qq = np.zeros((N,T))
log_w=np.zeros((N,T))
I = np.zeros((N,T))
xt = np.zeros((N,T))


# log_rec=np.zeros(T)

# % INIT: SAMPLE FROM THE PRIOR:
xu[:,0] = np.random.normal(0,1,size=N)

log_w[:, 0] = list(map(lambda x_u: stats.norm.pdf(y[0], 0, beta * np.exp(x_u)), xu[:,0])) 
# for i in range(N):
#     qq[i, 0] = stats.norm.pdf(y[0], m)
    
# normalize
qq[:, 0] = log_w[:, 0]
q[:, 0] = qq[:, 0] / sum(qq[:, 0])


I[:, 0] = multinomial_resample(q[:, 0])

I = I.astype(int)
x[:, 0] = xu[I[:, 0], 0].copy()
xt[:, 0] = x[:, 0].copy()


# print(x[:, 0])

for t in range(T-1):
    w = b * np.random.normal(size=N)
    xu[:, t+1] = a*x[:,t]+w
    
    d_t = beta * np.exp(x[:,t]/2)

    # compute weights and normalise
    qq[:,t+1] = list(map(lambda d: stats.norm.pdf(y[t+1], 0, d), d_t))
    q[:, t + 1] = qq[:, t + 1] / sum(qq[:, t + 1])
    I[:, t+1] = multinomial_resample(q[:, t+1])
    I = I.astype(int)
    #     % resampling
    x[:, t + 1] = xu[I[:, t+1], t + 1].copy() 
    xt[:, t+1] = x[:, t+1].copy()
    x[:, :t] = x[I[:, t+1], :t].copy()

    
    
    #%% q2.1.3 FFBSa
    
    # FFBSa
nn = N
idx = np.zeros((nn, T), dtype=int)
idx[:, -1] = multinomial_resample(q[:, -1])


for t in reversed(range(T-1)):
    w_f = np.zeros(nn)
    for i in range(nn):
        x_t = xu[idx[:,t], t]
        w_f[i] = q[i, t]*stats.norm.pdf(xu[i, t+1], rho*x_t[i], sigma)
    w_f = w_f/sum(w_f)
    idx[:, t] = multinomial_resample(w_f)
paths = [xu[idx[:, t], t] for t in range(T)]
paths = np.array(paths)

# plt.plot(paths, label="FFBSa", color="black")
# plt.plot(true_x, label="true", color="red")


#%% q2.1.3 a) plot

plt.plot(true_x, label="true", color="black", 
         alpha = 0.7,linestyle = "--")

plt.plot(sum(q*xt), label="(a) filter", linewidth=1)
plt.plot(sum(q*x), label="(a) smoother", linewidth=1)
plt.plot(np.mean(paths,1), label="(b) FFBSa", linewidth=1)
# plt.legend()
plt.xlabel("time n")
plt.ylabel("Expectation")
plt.title("Filter/Smoothing mean")
plt.legend(prop={'size': 9})
plt.savefig("plots/q2_1_3_a.pdf")

#%% q2.1.3 b)
plt.plot((sum(q*(xt*xt))-sum(q*xt)**2)/range(T), label="(a) filter")
plt.plot((sum(q*(x*x))-(sum(q*x)**2))/range(T), label="(a) smoother")
plt.plot((np.var(paths, 1))/range(T), label="(b) FFBSa")

plt.xlabel("time n")
plt.ylabel("Variance")
plt.title("Filter/Smoothing variance")
plt.legend(prop={'size': 10})
# plt.savefig("plots/q2_1_3_b.pdf")


#%% q2.1.4
# bias
plt.plot(np.abs(sum(q*xt)-np.mean(sum(q*x))),label="(a)filter")
plt.plot(np.abs(sum(q*x)-np.mean(sum(q*xt))),label="(a)smoother")
plt.plot(np.abs(np.mean(paths, axis=1) - np.mean(np.mean(paths, axis=1))), 
         label="(b)BFFSa")
plt.legend()


#%% q2.2


def un_log_norm_pdf(y, x):
    return(-0.5*np.log(2*np.pi)-np.log(sigma)-(y-x)**2/2*(sigma**2))

def volatility_SIR(RHO=0.91, N=50):
    # filtering approx
    x = np.zeros((N,T))
    # % resampled particles
    xu=np.zeros((N,T))
    # normalized weights
    q = np.zeros((N,T))
    # unormalized weights
    qq = np.zeros((N,T))

    log_w=np.zeros((N,T))
    I = np.zeros((N,T))

    log_rec=np.zeros(T)

    # % INIT: SAMPLE FROM THE PRIOR:
    xu[:,0] = np.random.normal(0,1,size=N)

    # log_w[:, 0] = list(map(lambda x_u: stats.norm.pdf(y[0], 0, beta * np.exp(x_u)), xu[:,0])) 
    for i in range(N):
        qq[i, 0] = stats.norm.pdf(y[0], 0, 0.5*np.exp(xu[i, 0]/2))

    q[:, 0] = qq[:, 0] / sum(qq[:, 0])

    # log_rec[0] = np.log(np.mean(qq[:, 0])) + offset

    # resample
    # I[:, 0] = np.random.choice(range(N), N, p=q[:, 0])
    # I[:, 0] = np.random.multinomial(N, q[:, 0], size=1)[0]
    # I[:, 0] = np.searchsorted(np.cumsum(q[:, 0]), random(N))
    # I[:, 0] = multinomial(q[:,0], N)
    I = multinomial_resample(q[:, 0])
    I = I.astype(int)
    x[:, 0] = xu[I, 0].copy()

    # print(x[:, 0])

    for t in range(T-1):
        w = b * np.random.normal(size=N)
        xu[:, t+1] = a*x[:,t]+w

        d_t = beta * np.exp(x[:,t]/2)

        # compute weights and normalise
    #     qq[:,t+1] = list(map(lambda d: stats.norm.pdf(y[t+1], 0, d), d_t))
        for i in range(N):
            qq[i, t+1] = stats.norm.pdf(y[t+1], 0, 0.5*np.exp(xu[i, t+1]/2))

        q[:, t + 1] = qq[:, t + 1] / sum(qq[:, t + 1])

    #     I[:, t+1] = np.random.choice(range(N), N, p=q[:, t+1])
    #     I[:, t+1] = np.random.multinomial(N, q[:, t+1], size=1)[0]
    #     I[:, t+1] = np.searchsorted(np.cumsum(q[:, t+1]), random(N))
    #     I[:, t+1] = multinomial(q[:,t+1], N)
        It = multinomial_resample(q[:, t+1])


        It = It.astype(int)

        #     % resampling
        x[:, t + 1] = xu[It, t + 1].copy()  
    #     print(I[:, t + 1])
        x[:, :t] = x[It, :t].copy()
    
    return(x)
#%%
r = 1

xr = volatility_SIR(RHO = r, N=N)
# print(xr[:, 10])

grads = np.zeros(N)
for i in range(N):
#         s = xr[:, p-1]*(xr[:, p] - r_k * xr[:, p-1])
#         grads += s
    for p in range(1,T):
        s = xr[i, p-1] * (xr[i, p] - r*xr[i,p-1])
#         print(s)
        grads[i] += s

grad = np.mean(grads)
grad/T

# xr[i, p-1] * (xr[i, p] - 0.8*xr[i,p-1])

#%%
# gradient ascent

step = 0.001
rho0_1 = 0.5
rho0_2 = -1
rho0_3 = 1.5
num_iter = 50
N=50

rho_list_1 = np.zeros(num_iter)
rho_list_1[0] = rho0_1

#%%

def gradient_asc(rho0, step=0.001):
    num_iter = 20
    N = 100
    rho_list = np.zeros(num_iter)
    rho_list[0] = rho0
    
    for k in range(num_iter-1):
        if (k%10)==0:
            print(k, rho_list[k])
        
        # compute gradient
        grad = 0
    
        # run PF
        r_k = rho_list[k]
        xr = volatility_SIR(RHO = r_k, N=N)
        grads = np.zeros(N)
        for i in range(N):
            for p in range(1,T):
                s = xr[i, p-1] * (xr[i, p] - rho_list[k]*xr[i,p-1])
                grads[i] += s
    
        
        grad = np.mean(grads)
        # update parameter
        rho_list[k+1] = r + step * grad
        print(grad, rho_list[k+1])
    return(rho_list)
        
    
#%%
rho_list_1 = gradient_asc(0.5)
rho_list_2 = gradient_asc(-1)
#%%
rho_list_3 = gradient_asc(1.5)
#%%
# print(np.mean(rho_list_1[:-10]))
print(rho_list_1[-1])
print(rho_list_2[-1])
print(rho_list_3[-1])

#%%

plt.plot(rho_list_1, 
         label=r"$\rho_0=0.5,\rho_{20}=0.8692$")
plt.plot(rho_list_2, 
         label=r"$\rho_0=-1,\rho_{20}=0.8606$")
plt.plot(rho_list_3, 
         label=r"$\rho_0=1.5,\rho_{20}=0.8950$")
plt.axhline(y=0.8, label =r"true $\rho=0.8$",
   color = "black", 
   alpha = 0.7,
   linestyle = "--")
plt.xlabel("k")
plt.ylabel(r"$\rho_k$")
plt.title(fr"step size $\gamma=${step}")
plt.legend()
plt.savefig("plots\q2_2.pdf")