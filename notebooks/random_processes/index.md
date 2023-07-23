# Random Processes

## ARIMA(p,d,q)

* **[adf_test.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/adf_test.ipynb)** The Augmented Dickey-Fuller (ADF) test is used to determine if an `AR(q)` process is stationary. Her the ADF test implementation is evaluated against simulated `AR(q)` processes.

* **[ar1_with_offset.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/ar1_with_offset.ipynb)** Test AR(1) simulation, parameter estimation and stationary mean and variance by comparing simulation parameters with estimation values and stationary mean and variance with cumulative values from simulations.

* **[ar1.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/ar1.ipynb)** Test `AR(p)` simulators by comparing first and second order moments of `AR(1)` computed from simulations with analytic results obtained by assuming stationarity.

* **[arima.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/arima.ipynb)** Test `ARIMA(p,d,q)` simulation and parameter estimation by comparing the parameters used in a simulation with the results obtained by parameter estimation.

* **[arma_order_estimation.ipynb](http://localhost:8888/files/notebooks/random_processes/arima/arma_order_estimation.ipynb?_xsrf=2%7Cdc5622b9%7C972d0332676f9814e33c8e21c7a0a95f%7C1642532355)** Test `ARMA(p,q)` simulations and order determination using autocorrelation function analysis and partial autocorrelation function analysis.

* **[arp_parameter_esimation.ipyn](http://localhost:8888/lab/tree/notebooks/random_processes/arima/arq_parameter_estimation.ipynb)** Test `AR(p)` parameter estimation using the Yule-Walker equations by comparing the parameters used in a simulation with the results obtained by parameter estimation.

* **[maq_parameter_estimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/maq_parameter_estimation.ipynb)** Test `MA(q)` parameter estimation by comparing the parameters used in a simulation with the results obtained by parameter estimation.

* **[maq.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima/maq.ipynb)** Test `MA(q)` simulator by comparing first and second order moments computed from simulations with analytic results.

## Brownian Motion

* **[bm_ensembles.ipyn](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/bm_ensembles.ipynb)** Test brownian motion and geometric brownian motion simulations by comparing mean and variance analytic results with calculations from ensembles.

* **[ecm_parameter_estimation.ipynb](http://localhost:65075/lab/tree/Develop/gly.fish/alef/notebooks/random_processes/brownian_motion/ecm_parameter_estimation.ipynb)** Test error correction model parameter estimation using simulation data.

* **[ecm_simulation.ipynb](http://localhost:65075/lab/tree/Develop/gly.fish/alef/notebooks/random_processes/brownian_motion/ecm_simulation.ipynb)** The error correction model models two cointegrated time series. Here the model is discussed and simulations are performed for a range of parameters.

* **[fbm_ensembles.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/fbm_ensembles.ipynb)** Test fractional brownian motion simulations using the FFT method by comparing analytic variance and autocorrelation results with calculations from ensembles.

* **[fbm_estimation_periodigram.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/fbm_estimation_periodigram.ipynb)** Test software implementing Hurst parameter, `H`, estimation using the periodigram method.

* **[fbm_estimation_variance_aggregation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/fbm_estimation_variance_aggregation.ipynb)** Test software implementing Hurst parameter, `H`, estimation using the variance aggregation method.

* **[fbm_expectations.ipynb](http://localhost:8888/lab/tree/notebooks/brownian_motion/random_processes/fbm_expectations.ipynb)** Plots of variance, correlation and autocorrelation for fractional brownian motion.

* **[fbm_variance_ratio_test.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/fbm_variance_ratio_test.ipynb)** Test implementation of the variance ratio test in determining if a time series is brownian motion. The test can also be used to determine if the fractional brownian motion Hurst parameter, `H`, satisfies `H < 1/2` or `H > 1/2`. The `H < 1/2` is used in the test for serial anti-correlation in a time series.

* **[fbm.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/brownian_motion/fbm.ipynb)** Examples of simulations using the Cholesky and FFT methods as the Hurst parameter, `H`, is varied.

## Ornstein Uhlenbeck Process

* **[ornstein_uhlenbeck_process.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/ornstein_uhlenbeck/ornstein_uhlenbeck_process.ipynb)** The Ornstein-Uhlenbeck stochastic differential equation describes a mean reverting random process. Analytic solutions or mean, variance, covariance and distribution are discussed and compared with simulations.

* **[ornstein_uhlenbeck_simulation.ipyn](http://localhost:8888/lab/tree/notebooks/random_processes/ornstein_uhlenbeck/ornstein_uhlenbeck_simulation.ipynb)** Simulations of the Ornstein-Uhlenbeck process are compared.

## VAR(n)

* **[var_multivariate_gaussian_noise.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/var/var_multivariate_gaussian_noise.ipynb)** The multivariate Gaussian distribution is the noise term in the VAR(n) model. Here, properties of the distribution are investigated.

* **[var_parameter_estimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/var/var_parameter_estimation.ipynb)** VAR is the generalization of the autoregressive process to multiple coupled time series. Parameter estimation is tested using simulation data.

* **[var_properties.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/var/var_properties.ipynb)** The results of VAR(1) and VAR(2) simulations are compared with stationary solutions for mean, variance and auto covariance obtained analytically.

* **[var_simulation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/var/var_simulation.ipynb)** The procedure used to simulate VAR processes is discussed.

## VEC(n)

* **[vecm_parameter_estimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/vecm/vecm_parameter_estimation.ipynb)** VECM is the generalization of the Error Correction Model (ECM) to an arbitrary number of cointegrated time series. Parameter estimation is tested using simulated data.

* **[vecm_parameter_simulation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/vecm/vecm_simulation.ipynb)** The VECM simulation procedure is discussed and parameter scan is performed.
