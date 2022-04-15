from enum import Enum
import numpy
from tabulate import tabulate

class VarianceRatioTestReport:
    def __init__(self, status, sig_level, test_type, s, statistics, p_values, critical_values):
        self.status = status
        self.sig_level = sig_level
        self.test_type = test_type
        self.s = s
        self.statistics = statistics
        self.p_values = p_values
        self.critical_values = critical_values

    def __repr__(self):
        f"VarianceRatioTestReport(status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values})"

    def __str__(self):
        return f"status={self.status}, sig_level={self.sig_level}, s={self.s}, statistics={self.statistics}, p_values={self.p_values}, critical_values={self.critical_values}"

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

    def table(self, tablefmt):
        header = self._header(tablefmt)
        result = self._results(tablefmt)
        return [header, result]

def adfuller_report(results, report, tablefmt):
    if not report:
        return
    stat = results[0]
    header = [["Test Statistic", stat],
              ["pvalue", results[1]],
              ["Lags", results[2]],
              ["Number Obs", results[3]]]
    status = ["Passed" if stat >= results[4][sig] else "Failed" for sig in ["1%", "5%", "10%"]]
    results = [["1%", results[4]["1%"], status[0]],
               ["5%", results[4]["5%"], status[1]],
               ["10%", results[4]["10%"], status[2]]]
    headers = ["Significance", "Critical Value", "Result"]
    print(tabulate(header, tablefmt=tablefmt))
    print(tabulate(results, tablefmt=tablefmt, headers=headers))
