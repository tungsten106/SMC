# SMC
A record for implementing Sequencial Monte Carlo algorithm with two models: Stochastic Volatility Model and Linear Gaussian Model.

For the first model we will perform particle smoothing using Forward Filtering Backward Sampling (FFBSa) to approximate the smoothing density $p(x_n| y_{0:T})$, and perform likelihood inference.
