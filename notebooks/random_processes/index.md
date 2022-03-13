# Random Processes

1. **ar1.ipynb** Test `AR(p)` simulators by comparing first and second order moments with analytic results.

2. **arma.ipynb** Test `ARMA(p,q)` simulations and parameter estimation using autocorrelation function analysis and partial autocorrelation function analysis.

3. **arq_parameter_esimation.ipynb** Test `AR(p)` parameter estimation using the Yule-Walker equations.

4. **bm_ensembles.ipynb** Test brownian motion and geometric brownian motion simulations by comparing mean and variance analytic results with calculations from ensembles.

4. **fbm_ensembles.ipynb** Test fractional brownian motion simulations using the FFT method by comparing analytic variance and autocorrelation results with calculations from ensembles.

5. **fbm_expectations.ipynb** Presentation of fractional brownian motion analytic results for variance, covariance and autocorrelation as a function of the Hurst parameter, `H`.

6. **fbm_parameter_estimation.ipynb** Test software implementing implementing Hurst parameter, `H`, estimation using the variance aggregation method and the periodigram method. Also, test implementation of the variance ratio test in determining if a time series has `H < 0.5` which indicates the series is anti-correlated.

7. **fbm.ipynb** Examples of simulations using the Cholesky and FFT methods as the Hurst parameter, `H`, is varied.
