import numpy
from enum import Enum
from datetime import datetime
from pandas import (DataFrame)

from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types)

from lib.data.est import (EstType, ParamEst, ARMAEst, OLSSingleVarEst,
                          create_estimates_from_dict, create_dict_from_estimates,
                          perform_est_for_type)
from lib.data.tests import (create_tests_from_dict, create_dict_from_tests)
from lib.data.schema import (DataType, DataSchema)

##################################################################################################################
# Meta Data Schema
class MetaData:
    def __init__(self, npts, data_type, params, desc, xlabel, ylabel, ests={}, tests={}, source_schema=None, formula=None):
        self.npts = npts
        self.schema = DataSchema.create(data_type)
        self.params = params
        self.desc = desc
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.source_schema = source_schema
        self.formula = formula
        self.ests = ests
        self.tests = tests
        self.data = {
          "npts": npts,
          "DataType": data_type,
          "Parameters": params,
          "Description": desc,
          "ylabel": ylabel,
          "xlabel": xlabel,
          "SourceSchema": source_schema,
          "Formula": formula,
          "Estimates": create_dict_from_estimates(ests),
          "Tests": create_dict_from_tests(tests)
        }

    def __repr__(self):
        return f"MetaData({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"npts=({self.npts}), " \
               f"schema=({self.schema}), " \
               f"params=({self.params}), " \
               f"desc=({self.desc}), " \
               f"xlabel=({self.xlabel}), " \
               f"ylabel=({self.ylabel}), " \
               f"ests=({self.ests}), " \
               f"tests=({self.tests}), " \
               f"source_schema=({self.source_schema}) " \
               f"formula=({self.formula}) "

    def insert_estimate(self, est):
        self.ests[est.key()] = est
        self.data["Estimates"][est.key()] = est.data

    def insert_test(self, test):
        self.ests[test.key()] = test
        self.data["Tests"][test.key()] = test.data

    def params_str(self):
        return MetaData.params_to_str(self.params)

    def get_data(self, df):
        return self.schema.get_data(df)

    def get_estimate(self, est_key):
        return self.ests[est_key]

    @staticmethod
    def from_dict(meta_data):
        source_schema =  meta_data["SourceSchema"] if "SourceSchema" in meta_data else None
        formula =  meta_data["Formula"] if "Formula" in meta_data else None

        if "Estimates" in meta_data:
            ests = create_estimates_from_dict(meta_data["Estimates"])
        else:
            ests = {}

        if "Tests" in meta_data:
            tests = create_tests_from_dict(meta_data["Tests"])
        else:
            tests = {}

        return MetaData(
            npts=meta_data["npts"],
            data_type=meta_data["DataType"],
            params=meta_data["Parameters"],
            desc=meta_data["Description"],
            xlabel=meta_data["xlabel"],
            ylabel=meta_data["ylabel"],
            ests=ests,
            tests=tests,
            source_schema=source_schema,
            formula=formula
        )

    @classmethod
    def get_data_type(cls, df, data_type):
        schema = DataSchema.create(data_type)
        return MetaData.get(df, schema)

    @classmethod
    def get(cls, df):
        schema = cls.get_schema(df)
        return MetaData.from_dict(df.attrs[schema.ycol])

    @classmethod
    def get_source_meta_data(cls, df):
        schema = cls.get_source_schema(df)
        if schema is not None:
            return MetaData.from_dict(df.attrs[schema.ycol])
        else:
            return None

    @classmethod
    def set(cls, df, meta_data):
        schema = cls.get_schema(df)
        df.attrs[schema.ycol]  = meta_data.data

    @classmethod
    def add_estimate(cls, df, est):
        meta_data = MetaData.get(df)
        meta_data.insert_estimate(est)
        MetaData.set(df, meta_data)

    @classmethod
    def add_test(cls, df, data_type, test):
        meta_data = MetaData.get_data_type(df, data_type)
        meta_data.insert_test(test)
        MetaData.set(df, data_type, meta_data)

    @classmethod
    def get_data(cls, df):
        schema = cls.get_schema(df)
        return schema.get_data(df)

    @classmethod
    def get_schema(cls, df):
        return df.attrs["Schema"]

    @classmethod
    def set_schema(cls, df, schema):
        df.attrs["Schema"] = schema

    @classmethod
    def get_source_schema(cls, df):
        return df.attrs["SourceSchema"]

    @classmethod
    def set_source_schema(cls, df, schema):
        df.attrs["SourceSchema"] = schema

    @classmethod
    def get_name(cls, df):
        return df.attrs["Name"]

    @classmethod
    def set_name(cls, df, name):
        df.attrs["Name"] = name

    @classmethod
    def get_date(cls, df):
        return df.attrs["Date"]

    @classmethod
    def set_date(cls, df):
        df.attrs["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    @classmethod
    def get_iterations(cls, df):
        return df.attrs["SchemaIterations"]

    @classmethod
    def set_iterations(cls, df, iter):
        df.attrs["SchemaIterations"] = iter

    @staticmethod
    def params_to_str(params):
        set_param_keys = []
        for key in params.keys():
            value = params[key]
            if isinstance(value, list) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, numpy.ndarray) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, int) and value != 0:
                set_param_keys.append(key)
            if isinstance(value, float) and value != 0.0:
                set_param_keys.append(key)
            if isinstance(value, str) and value != "":
                set_param_keys.append(key)
        if len(set_param_keys) == 0:
            return ""
        params_strs = []
        for key in set_param_keys:
            value = params[key]
            if value is None:
                continue
            if isinstance(value, numpy.ndarray):
                value_str = numpy.array2string(value, precision=2, separator=',', suppress_small=True)
            else:
                value_str = f"{value}"
            params_strs.append(f"{key}={value_str}")
        return ", ".join(params_strs)

##################################################################################################################
# Perform estimate
def perform_est(df, est_type, **kwargs):
    x, y = MetaData.get_data(df)
    result, est = perform_est_for_type(x, y, est_type, **kwargs)
    MetaData.add_estimate(df, est)
    return result
