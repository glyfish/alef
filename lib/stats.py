import numpy
from pandas import DataFrame
import statsmodels.api as sm
from enum import Enum

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema, create_schema)

class RegType(Enum):
    LINEAR = 1
    LOG = 2
    XLOG = 3
    YLOG = 4

def ensemble_mean(dfs, data_type=DataType.TIME_SERIES):
    if len(dfs) == 0:
        raise Exception(f"no data frames")
    nsim = len(dfs)
    x, samples = _samples_from_dfs(dfs, data_type)
    npts = len(x)
    mean = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            mean[i] += samples[j][i] / float(nsim)
    return _create_ensemble_avg_data_frame(x, mean, nsim, dfs[0], data_type, DataType.MEAN, "Ensemble Mean", r"$\mu$")

def ensemble_sd(dfs, data_type=DataType.TIME_SERIES):
    if len(dfs) == 0:
        raise Exception(f"no data frames")
    nsim = len(dfs)
    x, samples = _samples_from_dfs(dfs, data_type)
    mean_df = ensemble_mean(samples)
    _, mean = DataSchema.get_data_type(DataType.MEAN)
    npts = len(x)
    sd = numpy.zeros(npts)
    for i in range(npts):
        for j in range(nsim):
            sd[i] += (samples[j][i] - mean[i])**2 / float(nsim)
    return _create_ensemble_avg_data_frame(x, numpy.sqrt(sd), nsim, dfs[0], data_type, DataType.SD, "Ensemble SD", r"$\sigma$")

def ensemble_acf(dfs, nlags=None, data_type=DataType.TIME_SERIES):
    if len(dfs) == 0:
        raise Exception(f"no data frames")
    nsim = len(dfs)
    x, samples = _samples_from_dfs(dfs, data_type)
    if nlags is None or nlags > len(x):
        nlags = len(x)
    ac_avg = numpy.zeros(nlags)
    for j in range(nsim):
        ac = acf(samples[j], nlags).real
        for i in range(nlags):
            ac_avg[i] += ac[i]
    return _create_ensemble_avg_data_frame(x[:nlags], ac_avg/float(nsim), nsim, dfs[0], data_type, DataType.ACF, "Ensemble ACF", r"$\rho$")

def cumu_mean(y):
    ny = len(y)
    mean = numpy.zeros(ny)
    mean[0] = y[0]
    for i in range(1, ny):
        mean[i] = (float(i)*mean[i-1]+y[i])/float(i+1)
    return mean

def cumu_sd(y):
    mean = cumu_mean(y)
    ny = len(y)
    var = numpy.zeros(ny)
    var[0] = y[0]**2
    for i in range(1, ny):
        var[i] = (float(i)*var[i-1]+y[i]**2)/float(i+1)
    return numpy.sqrt(var-mean**2)

def cumu_cov(x, y):
    nsample = min(len(x), len(y))
    cov = numpy.zeros(nsample)
    meanx = cumu_mean(x)
    meany = cumu_mean(y)
    cov[0] = x[0]*y[0]
    for i in range(1, nsample):
        cov[i] = (float(i)*cov[i-1]+x[i]*y[i])/float(i+1)
    return cov-meanx*meany

def cov(x, y):
    nsample = len(x)
    meanx = numpy.mean(x)
    meany = numpy.mean(y)
    c = 0.0
    for i in range(nsample):
        c += x[i]*y[i]
    return c/nsample-meanx*meany

def agg(samples, m):
    n = len(samples)
    d = int(n/m)
    agg = numpy.zeros(d)
    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k]/m
    return agg

def agg_var(samples, m_vals):
    npts = len(m_vals)
    agg_var = numpy.zeros(npts)
    for i in range(npts):
        m = int(m_vals[i])
        agg_vals = agg(samples, m)
        agg_mean = numpy.mean(agg_vals)
        d = len(agg_vals)
        for k in range(d):
            agg_var[i] += (agg_vals[k] - agg_mean)**2/(d - 1)
    return agg_var

def pspec(x):
    n = len(x)
    μ = x.mean()
    x_shifted = x - μ
    energy = numpy.sum(x_shifted**2)
    x_padded = numpy.concatenate((x_shifted, numpy.zeros(n-1)))
    x_fft = numpy.fft.fft(x_padded)
    power = numpy.conj(x_fft)*x_fft
    return power[1:n].real/(n*energy)

def pdf_hist(samples, range, nbins=50):
    return numpy.histogram(samples, bins=nbins, range=range, density=True)

def cdf_hist(x, pdf):
    npoints = len(pdf)
    cdf = numpy.zeros(npoints)
    for i in range(npoints):
        dx = x[i+1] - x[i]
        cdf[i] = numpy.sum(pdf[:i])*dx
    return cdf

def acf(samples, nlags):
    return sm.tsa.stattools.acf(samples, nlags=nlags, fft=True)

## OLS
def OLS(y, x, type=RegType.LINEAR):
    if type == RegType.LOG:
        x = numpy.log10(x)
        y = numpy.log10(y)
    x = sm.add_constant(x)
    return sm.OLS(y, x)

def OLS_fit(y, x, type=RegType.LINEAR):
    model = OLS(y, x, type=type)
    results = model.fit()
    results.summary()
    return results

## Private
def _samples_from_dfs(dfs, data_type):
    schema = create_schema(data_type)
    samples = []
    for df in dfs:
        x, y = schema.get_data(df)
        samples.append(y)
    return x, samples

def _create_ensemble_avg_data_frame(x, y, nsim, df, source_data_type, data_type, desc, ylabel):
    source_schema = create_schema(source_data_type)
    source_meta_data = MetaData.get(df, source_schema)
    source_meta_data["Parameters"]["nsim"] = nsim
    meta_data = MetaData(
        npts=len(y),
        data_type=data_type,
        params=source_meta_data.params,
        desc=f"{source_meta_data.desc} {desc}",
        xlabel=source_meta_data.xlabel,
        ylabel=ylabel,
    )
    return DataSchema.create_data_frame(x, y, meta_data)
