from enum import Enum
from pandas import DataFrame
import uuid
import numpy

from lib import stats
from lib.models import fbm
from lib.models import bm
from lib.models import arima
from lib.models import adf
from lib.models import ou

from lib.data.meta_data import (MetaData)
from lib.data.schema import (DataType, DataSchema)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space)

##################################################################################################################
# Specify Data Source Types used in analysis
#
# xcol: name of data doamin in DataFrame
# ycol: name of data range in DataFrame
#
# f: Function used to compute ycol from xcol, fy is assumed to have the form
#     f(x) -> ycol
#
class SourceBase(Enum):
    def create(self, **kwargs):
        x = get_param_default_if_missing("x", None, **kwargs)
        if x is None:
            x = create_space(**kwargs)
        source = self._create_data_source(x, **kwargs)
        return source.create()

    def create_parameter_scan(self, *args):
        dfs = []
        for kwargs in args:
            dfs.append(self.create(**kwargs))
        return dfs

    def create_ensemble(self, nsim, **kwargs):
        ensemble = []
        for i in range(nsim):
            ensemble.append(self.create(**kwargs))
        return ensemble

    def _create_data_source(self, x, **kwargs):
        Exception(f"_create_data_source not implemented")

###################################################################################################
# DataSource is used to create input data types. DataTypes can be model simulations
# or real data. The model properies define the meta data of the source and a Function
# of the form f(x) used to create the source. The create() method returns a DataFrame()
# conmtining the DataType.
#
class DataSource:
    def __init__(self, source_type, schema, name, params, ylabel, xlabel, desc, f, x=None):
        self.source_type = source_type
        self.schema = schema
        self.name = name
        self.params = params
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.desc = desc
        self.f = f
        self.x = x

    def __repr__(self):
        return f"DataSource({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"desc=({self.desc})"

    def meta_data(self, npts):
        return MetaData(
            npts=npts,
            data_type=self.schema.data_type,
            params=self.params,
            desc=self.desc,
            xlabel=self.xlabel,
            ylabel=self.ylabel
        )

    def create(self):
        y = self.f(self.x)
        df = self.schema.create_data_frame(self.x, y)
        DataSchema.set_source_type(df, None)
        DataSchema.set_source_name(df, None)
        DataSchema.set_source_schema(df, None)
        DataSchema.set_date(df)
        DataSchema.set_type(df, self.source_type)
        DataSchema.set_name(df, self.name)
        DataSchema.set_schema(df, self.schema)
        DataSchema.set_iterations(df, None)
        MetaData.set(df, self.meta_data(len(y)))
        return df
