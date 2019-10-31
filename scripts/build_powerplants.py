# coding: utf-8
"""
Retrieves conventional powerplant capacities and locations from `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_, assigns these to buses and creates a ``.csv`` file.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter: 
      custom_powerplants: 

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: ../img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

"""

import logging
from scipy.spatial import cKDTree as KDTree

import pypsa
import powerplantmatching as pm
import pandas as pd

logger = logging.getLogger(__name__)


def add_custom_powerplants(ppl):
    custom_ppl_query = snakemake.config['electricity']['custom_powerplants']
    if not custom_ppl_query:
        return ppl
    add_ppls = pd.read_csv(snakemake.input.custom_powerplants, index_col=0)
    if isinstance(custom_ppl_query, str):
        add_ppls.query(add_ppls, inplace=True)
    return ppl.append(add_ppls, sort=False)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict

        snakemake = MockSnakemake(
            input=Dict(base_network='networks/base.nc',
                       custom_powerplants='data/custom_powerplants.csv'),
            output=['resources/powerplants.csv']
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.base_network)
    countries = n.buses.country.unique()

    ppl = (pm.powerplants(from_url=True)
           .powerplant.convert_country_to_alpha2()
           .query('Fueltype not in ["Solar", "Wind"] and Country in @countries')
           .replace({'Technology': {'Steam Turbine': 'OCGT'}})
            .assign(Fueltype=lambda df: (
                    df.Fueltype
                      .where(df.Fueltype != 'Natural Gas',
                             df.Technology.replace('Steam Turbine',
                                                   'OCGT').fillna('OCGT')))))

    ppl_query = snakemake.config['electricity']['powerplants_filter']
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)

    ppl = add_custom_powerplants(ppl) # add carriers from own powerplant files

    cntries_without_ppl = [c for c in countries if c not in ppl.Country.unique()]

    substation_i = n.buses.query('substation_lv').index
    kdtree = KDTree(n.buses.loc[substation_i, ['x','y']].values)

    ppl['bus'] = substation_i[kdtree.query(ppl[['lon','lat']].values)[1]]

    if cntries_without_ppl:
        logging.warning(f"No powerplants known in: {', '.join(cntries_without_ppl)}")

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning(f"Couldn't find close bus for {bus_null_b.sum()} powerplants")

    ppl.to_csv(snakemake.output[0])
