# Random Processes

* **[adf_test.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/adf_test.ipynb)** The Augmented Dickey-Fuller (ADF) test is used to determine if an `AR(q)` process is stationary. Her the ADF test implementation is evaluated against simulated `AR(q)` processes.

* **[ar1_with_offset.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/ar1_with_offset.ipynb)** Test AR(1) simulation, parameter estimation and stationary mean and variance by comparing simulation parameters with estimation values and stationary mean and variance with cumulative values from simulations.

* **[ar1.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/ar1.ipynb)** Test `AR(p)` simulators by comparing first and second order moments of `AR(1)` computed from simulations with analytic results obtained by assuming stationarity.

* **[arima.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arima.ipynb)** Test `ARIMA(p,d,q)` simulation and parameter estimation by comparing the parameters used in a simulation with the results obtained by parameter estimation.

* **[arma_order_estimation.ipynb](http://localhost:8888/files/notebooks/random_processes/arma_order_estimation.ipynb?_xsrf=2%7Cdc5622b9%7C972d0332676f9814e33c8e21c7a0a95f%7C1642532355)** Test `ARMA(p,q)` simulations and order determination using autocorrelation function analysis and partial autocorrelation function analysis.

* **[arp_parameter_esimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/arq_parameter_estimation.ipynb)** Test `AR(p)` parameter estimation using the Yule-Walker equations by comparing the parameters used in a simulation with the results obtained by parameter estimation..

* **[bm_ensembles.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/bm_ensembles.ipynb)** Test brownian motion and geometric brownian motion simulations by comparing mean and variance analytic results with calculations from ensembles.

* **[fbm_ensembles.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm_ensembles.ipynb)** Test fractional brownian motion simulations using the FFT method by comparing analytic variance and autocorrelation results with calculations from ensembles.

* **[fbm_estimation_periodigram.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm_estimation_periodigram.ipynb)** Test software implementing Hurst parameter, `H`, estimation using the periodigram method.

* **[fbm_estimation_variance_aggregation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm_estimation_variance_aggregation.ipynb)** Test software implementing Hurst parameter, `H`, estimation using the variance aggregation method.

* **[fbm_expectations.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm_expectations.ipynb)** Plots of variance, correlation and autocorrelation for fractional brownian motion.

* **[fbm_variance_ratio_test.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm_variance_ratio_test.ipynb)** Test implementation of the variance ratio test in determining if a time series is brownian motion. The test can also be used to determine if the fractional brownian motion Hurst parameter, `H`, satisfies `H < 1/2` or `H > 1/2`. The `H < 1/2` is used in the test for serial anti-correlation in a time series.

* **[fbm.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/fbm.ipynb)** Examples of simulations using the Cholesky and FFT methods as the Hurst parameter, `H`, is varied.

* **[maq_parameter_estimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/maq_parameter_estimation.ipynb)** Test `MA(q)` parameter estimation by comparing the parameters used in a simulation with the results obtained by parameter estimation.

* **[maq.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/maq.ipynb)** Test `MA(q)` simulator by comparing first and second order moments computed from simulations with analytic results.

* **[ornstein_uhlenbeck_parameter_estimation.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/ornstein_uhlenbeck_parameter_estimation.ipynb)** Ornstein-Uhlenbeck process parameters parameters are estimated from simulations.

* **[ornstein_uhlenbeck_process.ipynb](http://localhost:8888/lab/tree/notebooks/random_processes/ornstein_uhlenbeck_process.ipynb)** The Ornstein-Uhlenbeck stochastic differential equation describes a mean reverting random process. Analytic solutions or mean, variance, covariance and distribution are discussed and compared with simulations.

* **[ornstein_uhlenbeck_simulation.ipyn](http://localhost:8888/lab/tree/notebooks/random_processes/ornstein_uhlenbeck_simulation.ipynb)** Simulations of the Ornstein-Uhlenbeck process are compared.
