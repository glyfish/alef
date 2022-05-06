import numpy
from enum import Enum
from pandas import (DataFrame)

from lib.data.est import (create_esimates_from_dict, create_dict_from_estimates)
from lib.data.tests import (create_tests_from_dict, create_dict_from_tests)
from lib.data.schema import (DataType, create_schema)

##################################################################################################################
# Meta Data Schema
class MetaData:
    def __init__(self, npts, data_type, params, desc, xlabel, ylabel, ests={}, tests={}, source_schema=None):
        self.npts = npts
        self.schema = create_schema(data_type)
        self.params = params
        self.desc = desc
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.source_schema = source_schema
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
          "Estimates": create_dict_from_estimates(ests),
          "Tests": create_dict_from_tests(tests)
        }

    def __repr__(self):
        return f"MetaData({self._props()})"

    def __str__(self):
        return self._props()

    def _props(self):
        return f"npts=({self.npts}), schema=({self.schema}), params=({self.params}), desc=({self.desc}), xlabel=({self.xlabel}), ylabel=({self.ylabel}), ests=({self.ests}), tests=({self.tests}), source_schema=({self.source_schema})"

    def insert_estimate(self, est):
        self.ests[est.key()] = est
        self.data["Estimates"][est.key()] = est.data

    def insert_test(self, test):
        self.ests[test.key()] = test
        self.data["Tests"][test.key()] = test.data

    def params_str(self):
        params_keys = self.params
        set_param_keys = []
        for key in params_keys:
            value = self.params[key]
            if isinstance(value, list) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, numpy.ndarray) and len(value) > 0:
                set_param_keys.append(key)
            if isinstance(value, int) and value > 0:
                set_param_keys.append(key)
            if isinstance(value, float) and value > 0.0:
                set_param_keys.append(key)
            if isinstance(value, str) and value != "":
                set_param_keys.append(key)
        if len(set_param_keys) == 0:
            return ""
        params_strs = []
        for key in set_param_keys:
            value = self.params[key]
            if isinstance(value, numpy.ndarray):
                value_str = numpy.array2string(value, precision=2, separator=',', suppress_small=True)
            else:
                value_str = f"{value}"
            params_strs.append(f"{key}={value_str}")
        return ", ".join(params_strs)

    @staticmethod
    def from_dict(meta_data):
        source_schema =  meta_data["SourceSchema"] if "SourceSchema" in meta_data else None

        if "Estimates" in meta_data:
            ests = create_esimates_from_dict(meta_data["Estimates"])
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
            source_schema=source_schema
        )

    @staticmethod
    def get_data_type(df, data_type):
        schema = create_schema(data_type)
        return MetaData.get(df, schema)

    def get(df, schema):
        return MetaData.from_dict(df.attrs[schema.ycol])

    @staticmethod
    def set(df, data_type, meta_data):
        schema = create_schema(data_type)
        df.attrs[schema.ycol]  = meta_data.data

    @staticmethod
    def add_estimate(df, data_type, est):
        meta_data = MetaData.get(df, data_type)
        meta_data.insert_estimate(est)
        MetaData.set(df, data_type, meta_data)

    def add_test(df, data_type, test):
        meta_data = MetaData.get(df, data_type)
        meta_data.insert_test(test)
        MetaData.set(df, data_type, meta_data)
