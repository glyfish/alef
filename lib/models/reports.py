from tabulate import tabulate

##################################################################################################################
# Variance ratio test report
##################################################################################################################
class VarianceRatioTestReport:
    def __init__(self, sig_level, test_type, s, stats, p_values, critical_values):
        self.sig_level = sig_level
        self.test_type = test_type
        self.s = s
        self.stats = stats
        self.p_values = p_values
        self.critical_values = critical_values

    def __repr__(self):
        return f"VarianceRatioTestReport({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"status={self.status}, " \
               f"sig_level={self.sig_level}, " \
               f"s={self.s}, " \
               f"statistics={self.statistics}, " \
               f"p_values={self.p_values}, " \
               f"critical_values={self.critical_values}"

    def _header(self, tablefmt):
        test_status = "Passed" if self.status else "Failed"
        header = [["Result", test_status], ["Test Type", self.test_type], ["Significance", f"{int(100.0*self.sig_level)}%"]]
        if self.critical_values[0] is not None:
            header.append(["Lower Critical Value", format(self.critical_values[0], '1.3f')])
        if self.critical_values[1] is not None:
            header.append(["Upper Critical Value", format(self.critical_values[1], '1.3f')])
        return tabulate(header, tablefmt=tablefmt)

    def _results(self, tablefmt):
        if self.critical_values[0] is None:
            z_result = [self.statistics[i] < self.critical_values[1] for i in range(len(self.statistics))]
        elif self.critical_values[1] is None:
            z_result = [self.statistics[i] > self.critical_values[0] for i in range(len(self.statistics))]
        else:
            z_result = [self.critical_values[1] > self.statistics[i] > self.critical_values[0] for i in range(len(self.statistics))]
        z_result = ["Passed" if zr else "Failed" for zr in z_result]
        s_result = [int(s_val) for s_val in self.s]
        stat_result = [format(stat, '1.3f') for stat in self.statistics]
        pval_result = [format(pval, '1.3f') for pval in self.p_values]
        results = [s_result]
        results.append(stat_result)
        results.append(pval_result)
        results.append(z_result)
        results = numpy.transpose(numpy.array(results))
        return tabulate(results, headers=["s", "Z(s)", "pvalue", "Result"], tablefmt=tablefmt)

    def _table(self, tablefmt):
        header = self._header(tablefmt)
        result = self._results(tablefmt)
        return [header, result]

    def summary(self, tablefmt="fancy_grid"):
        if not report:
            return
        table = self.table(tablefmt)
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
