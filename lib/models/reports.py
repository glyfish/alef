from tabulate import tabulate
import numpy

##################################################################################################################
# Variance ratio test report
##################################################################################################################
class VarianceRatioTestReport:
    def __init__(self, sig_level, hyp_type, s_vals, stats, p_vals, critical_values):
        self.sig_level = sig_level
        self.hyp_type = hyp_type
        self.s_vals = s_vals
        self.stats = stats
        self.p_vals = p_vals
        self.critical_values = critical_values
        if self.critical_values[0] is None:
            self.status_vals = [self.stats[i] < self.critical_values[1] for i in range(len(self.stats))]
        elif self.critical_values[1] is None:
            self.status_vals = [self.stats[i] > self.critical_values[0] for i in range(len(self.stats))]
        else:
            self.status_vals = [self.critical_values[1] > self.stats[i] > self.critical_values[0] for i in range(len(self.stats))]

    def __repr__(self):
        return f"VarianceRatioTestReport({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status_vals={self.status_vals}, " \
               f"sig_level={self.sig_level}, " \
               f"s_vals={self.s_vals}, " \
               f"stats={self.stats}, " \
               f"p_vals={self.p_vals}, " \
               f"critical_values={self.critical_values}"

    def _header(self, tablefmt):
        header = [["Hypothesis Type", self.hyp_type], ["Significance", f"{int(100.0*self.sig_level)}%"]]
        if self.critical_values[0] is not None:
            header.append(["Lower Critical Value", format(self.critical_values[0], '1.3f')])
        if self.critical_values[1] is not None:
            header.append(["Upper Critical Value", format(self.critical_values[1], '1.3f')])
        return tabulate(header, tablefmt=tablefmt)

    def _results(self, tablefmt):
        status_result = ["Passed" if status else "Failed" for status in self.status_vals]
        s_result = [int(s_val) for s_val in self.s_vals]
        stat_result = [format(stat, '1.3f') for stat in self.stats]
        pval_result = [format(pval, '1.3f') for pval in self.p_vals]
        results = [s_result]
        results.append(stat_result)
        results.append(pval_result)
        results.append(status_result)
        results = numpy.transpose(numpy.array(results))
        return tabulate(results, headers=["s", "Z(s)", "pvalue", "Result"], tablefmt=tablefmt)

    def _table(self, tablefmt):
        header = self._header(tablefmt)
        result = self._results(tablefmt)
        return [header, result]

    def summary(self, tablefmt="fancy_grid"):
        table = self._table(tablefmt)
        print(table[0])
        print(table[1])

##################################################################################################################
# ADF test report
##################################################################################################################
class ADFTestReport:
    def __init__(self, result):
        self.stat = result[0]
        self.pval = result[1]
        self.lags = result[2]
        self.nobs = result[3]
        self.sig_str = ["1%", "5%", "10%"]
        self.sig = [0.01, 0.05, 0.1]
        self.critical_vals = [result[4][sig] for sig in self.sig_str]
        self.status_vals = [self.stat >= val for val in self.critical_vals]
        self.status_str = ["Passed" if status else "Failed" for status in self.status_vals]

    def summary(self, tablefmt="fancy_grid"):
        headers = ["Significance", "Critical Value", "Result"]
        header = [["Test Statistic", self.stat],
                  ["pvalue", self.pval],
                  ["Lags", self.lags],
                  ["Number Obs", self.nobs]]
        results = [[self.sig_str[i], self.critical_vals[i], self.status_str[i]] for i in range(3)]
        print(tabulate(header, tablefmt=tablefmt))
        print(tabulate(results, tablefmt=tablefmt, headers=headers))

##################################################################################################################
# Ornstein-Uhulenbeck processs parameter estimate report
##################################################################################################################
class OUReport:
    def __init__(self, result, Δt, x0):
        conf_int = results.conf_int()
        self.delta_t = Δt
        self.x0 = x0
        self._offset_est = result.params.iloc[0]
        self._offset_error = result.bse.iloc[0]
        self._coeff_est = result.params.iloc[1]
        self._coeff_error = result.bse.iloc[1]
        self._sigma2_est = result.params.iloc[2]
        self._sigma2_error = result.bse.iloc[2]

    def mu_est(self):
        return self._offset_est/(1.0 - self._coeff_est)

    def mu_error(self):
        return (self._offset_est*self._coeff_error + self._offset_error)/(1.0 - self._coeff_est)

    def lambda_est(self):
        return -numpy.log(self._coeff_est)/self.delta_t

    def lambda_error(self):
        return -self._coeff_error/(self._coeff_est*self.delta_t)

    def sigma2_est(self):
        return 2.0*self.lambda_est()*self._sigma2_est/(1.0 - self._coeff_est**2)

    def sigma2_error(self):
        return 4.0*self.lambda_est()*self._coeff_est*self._sigma2_est*self._coeff_error/(1.0 - self._coeff_est**2)**2 + \
               2.0*self._sigma2_est*self.lambda_error()/(1.0 - self._coeff_est**2) + \
               2.0*self.lambda_est()*self._sigma2_error/(1.0 - self._coeff_est**2)

    def summary(self, tablefmt="fancy_grid"):
        header = [["Δt", self.delta_t],
                  ["X0", self.x0]]
        headers = ["Parameter", "Estimate", "Error"]
        results = [["μ", self.mu_est(), self.mu_error()],
                   ["λ", self.lambda_est(), self.lambda_error()],
                   ["σ2", self.sigma2_est(), self.sigma2_error()]]
        print(tabulate(header, tablefmt=tablefmt))
        print(tabulate(results, tablefmt=tablefmt, headers=headers))
