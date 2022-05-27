# Introduction of the contents
This project is implementing Sequencial Monte Carlo simulations and inferences on two models(stochastic volatility model and linear Gaussian model). The project itself is part of my MSc study on Advanced simulation methods. The lecturer is Nicolas Kantas, who has given the idea of the whole concept and part of the code written in MATLAB.



## Stochastic Volatility Model
$$X_{n}=\rho X_{n-1}+\sigma V_{n}, \quad Y_{n}=\beta \exp \left(\frac{X_{n}}{2}\right) W_{n}$$
where $W_{n}, V_{n} \stackrel{\text { iid }}{\sim} \mathcal{N}(0,1), X_{0} \sim \mathcal{N}(0,1)$.
We run for $T=100$, with parameter values $\rho=$ $0.91, \sigma=1, \beta=0.5$.

For the first model we will perform particle smoothing using Forward Filtering Backward Sampling (FFBSa) to approximate the smoothing density $p(x_n| y_{0:T})$, and the likelihood inference on $\nabla_{\theta} \log p_{\theta}\left(y_{:: T}\right)$.

The code of implementation is given in `Stochastic_Volatility_Model.py`


## Linear Gaussian Model
$$X_{n}=\rho X_{n-1}+\tau V_{n}, \quad Y_{n}=X_{n}+\sigma W_{n}$$
where $W_{n}, V_{n} \stackrel{\text { iid }}{\sim} \mathcal{N}(0,1), X_{0} \sim \mathcal{N}(0,1)$. \\
We run for $T=100$, with parameter $\theta=(\rho, \tau, \sigma)=(0.8, 1, 0.5)$.

Here we want to inference for $p\left(\rho, \tau^{2} \mid y_{0: T}\right)$ assuming $\sigma=0.5$ is known.

The code of implementation is given in `Linear_Gaussian_Model.py`

