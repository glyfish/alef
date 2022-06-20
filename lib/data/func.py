from enum import Enum
from pandas import DataFrame
from datetime import datetime
import uuid
import numpy

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# DataFunc consist of the input schema and function used to compute resulting data columns
class FuncBase(str, Enum):
    def create(self, **kwargs):
        x = get_param_default_if_missing("x", None, **kwargs)
        if x is None:
            x = create_space(**kwargs)
        kwargs["npts"] = len(x)
        data_func = self._create_func(**kwargs)
        return data_func.create(x)

    def create_parameter_scan(self, *args):
        dfs = []
        for kwargs in args:
            dfs.append(self.create(**kwargs))
        return dfs

    def apply(self, df, **kwargs):
        data_func = self._create_func(**kwargs)
        return data_func.apply(df)

    def apply_to_list(self, dfs, **kwargs):
        data_func = self._create_func(**kwargs)
        return data_func.apply_list(dfs)

    def apply_parameter_scan(self, df, *args):
        dfs = []
        for kwargs in args:
            dfs.append(self.apply(df, **kwargs))
        return dfs

    def apply_to_ensemble(self, dfs, **kwargs):
        data_func = self._create_ensemble_data_func(**kwargs)
        return data_func.apply_ensemble(dfs)

    def _create_func(self, **kwargs):
        Exception(f"_create_func not implemented")

    def _create_ensemble_data_func(self, **kwargs):
        Exception(f"_create_ensemble_data_func not implemented")

###################################################################################################
# DataFunc consist of the input schema and function used to compute resulting data columns
#
# xcol: name of data domain in DataFrame
# ycol: name of data range in DataFrame
#
# fy: Function used to compute ycol from source DataType, fy is assumed to have the form
#     fy(fx(x),y) -> ycol
# fx: Function used to compute xcol from source DataType xcol, fx is assumed to have the form
#     fx(x) -> xcol
# fyx: Function used to compute both xcol and ycol from source data. fxy is used only if fy is
#      None. fyx is assumed to have the form fyx(x, y) -> ycol, x', fx(x') -> xcol
# source: DataType input into f used to compute xcol and ycol
#
class DataFunc:
    def __init__(self, func_type, data_type, source_type, params, ylabel, xlabel, desc, formula=None, fy=None, fx=None, fyx=None):
        self.func_type = func_type
        self.schema = data_type.schema()
        self.params = params
        self.source_schema = source_type.schema()
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.desc = desc
        self.formula = formula
        self.fy = fy
        self.fyx = fyx
        if fx is None:
            self.fx = lambda x: x
        else:
            self.fx = fx

    def __repr__(self):
        return f"DataFunc({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"func_type={self.func_type},  " \
               f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"source_schema=({self.source_schema}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"desc=({self.desc}), " \
               f"formula=({self.formula})"

    def _name(self):
        return f"{self.func_type.value}-{str(uuid.uuid4())}"

    def _set_df_meta_data(self, df, source_type, source_name):
        DataSchema.set_source_type(df, source_type)
        DataSchema.set_source_name(df, source_name)
        DataSchema.set_source_schema(df, self.source_schema)
        DataSchema.set_date(df)
        DataSchema.set_type(df, self.func_type)
        DataSchema.set_name(df, self._name())
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_iterations(df, None)

    def apply(self, df):
        x, y = self.source_schema.get_data(df)
        if self.fy is not None:
            x_result = self.fx(x)
            y_result = self.fy(x_result, y)
        elif self.fyx is not None:
            y_result, x_result = self.fyx(x, y)
            x_result = self.fx(x_result)
        else:
            Exception(f"fy or fyx must be specified")
        df_result = self.create_data_frame(x_result, y_result)
        source_name = DataSchema.get_source_name(df)
        source_type = DataSchema.get_source_type(df)
        self._set_df_meta_data(df_result, source_type, source_name)
        MetaData.set(df_result, self.meta_data(len(y_result)))
        df_result.attrs = df.attrs | df_result.attrs
        return df_result

    def apply_ensemble(self, dfs):
        if len(dfs) == 0:
            Exception(f"No DataFrames provided")
        x, y = self.source_schema.get_data_from_list(dfs)
        x_result = self.fx(x)
        y_result = self.fy(x_result, y)
        df = self.create_data_frame(x_result, y_result)
        source_name = DataSchema.get_name(dfs[0])
        source_type = DataSchema.get_type(dfs[0])
        self._set_df_meta_data(df, source_type, source_name)
        MetaData.set(df, self.meta_data(len(y_result)))
        df.attrs = dfs[0].attrs | df.attrs
        return df

    def apply_list(self, dfs):
        return [self.apply(df) for df in dfs]

    def meta_data(self, npts):
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            source_schema = self.source_schema,
            formula = self.formula
        )

    def ensemble_meta_data(self, npts, df):
        source_meta_data = MetaData.get(df)
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=source_meta_data.params | self.params,
            desc=f"{source_meta_data.desc} {self.desc}",
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            source_schema = None,
            formula = self.formula
        )

    def create(self, x):
        y = self.fy(x, None)
        df = self.create_data_frame(x, y)
        DataSchema.set_source_type(df, None)
        DataSchema.set_source_name(df, None)
        DataSchema.set_source_schema(df, self.source_schema)
        DataSchema.set_date(df)
        DataSchema.set_type(df, self.func_type)
        DataSchema.set_name(df, self._name())
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_iterations(df, None)
        MetaData.set(df, self.meta_data(len(y)))
        return df

    def create_data_frame(self, x, y):
        return self.schema.create_data_frame(x, y)

###################################################################################################
## Helpers
def _get_s_vals(**kwargs):
    linear = get_param_default_if_missing("linear", False, **kwargs)
    s_min = get_param_default_if_missing("s_min", 1.0, **kwargs)
    npts = get_param_default_if_missing("npts", None, **kwargs)
    s_max = get_param_default_if_missing("s_max", None, **kwargs)
    s_vals = get_param_default_if_missing("s_vals", None, **kwargs)
    if npts is not None and s_max is not None:
        if linear:
            return create_space(npts=npts, xmax=s_max, xmin=s_min)
        else:
            return create_logspace(npts=npts, xmax=s_max, xmin=s_min)
    elif s_vals is not None:
        return s_vals
    else:
        raise Exception(f"s_max and npts or s_vals is required")
