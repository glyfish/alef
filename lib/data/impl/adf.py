from enum import Enum
import uuid
import numpy

from lib.models import adf

from lib.data.source import (DataSource, SourceBase)
from lib.data.schema import (DataType)
from lib.utils import (get_param_throw_if_missing, get_param_default_if_missing,
                       verify_type, verify_types, create_space, create_logspace)

###################################################################################################
# Create ADF Functions
class ADF:
    # Source
    class Source(SourceBase):
        DF = "DF"          # Dickey-Fuller distribution simulation

        def _create_data_source(self, x, **kwargs):
            return _create_data_source(self, x, **kwargs)

###################################################################################################
## Create DataSource object for specified type
###################################################################################################
def _create_data_source(source_type, x, **kwargs):
    if source_type.value == ADF.Source.DF.value:
        return _create_df_source(source_type, x, **kwargs)
    else:
        raise Exception(f"Source type is invalid: {source_type}")

###################################################################################################
# Source.DF:
def _create_df_source(source_type, x, **kwargs):
    nstep = get_param_default_if_missing("nstep", 100, **kwargs)
    nsim = get_param_default_if_missing("nsim", 1000, **kwargs)
    f = lambda x : adf.dist_ensemble(nstep, nsim)
    return DataSource(source_type=source_type,
                      schema=DataType.TIME_SERIES.schema(),
                      name=f"Dickey-Fuller-Simulation-{str(uuid.uuid4())}",
                      params={"nstep": nstep, "nsim": nsim},
                      ylabel=r"$S_t$",
                      xlabel=r"$t$",
                      desc=f"Dickey-Fuller Distribution",
                      f=f,
                      x=x)
