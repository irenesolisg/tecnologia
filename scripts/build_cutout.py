"""
Create cutouts with `atlite <https://atlite.readthedocs.io/en/latest/>`_.

For this rule to work you must have

- installed the `Copernicus Climate Data Store <https://cds.climate.copernicus.eu>`_ ``cdsapi`` package  (`install with `pip``) and
- registered and setup your CDS API key as described `on their website <https://cds.climate.copernicus.eu/api-how-to>`_.

.. seealso::
    For details on the weather data read the `atlite documentation <https://atlite.readthedocs.io/en/latest/>`_.
    If you need help specifically for creating cutouts `the corresponding section in the atlite documentation <https://atlite.readthedocs.io/en/latest/examples/create_cutout.html>`_ should be helpful.

Relevant Settings
-----------------

.. code:: yaml

    atlite:
        nprocesses:
        cutouts:
            {cutout}:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`atlite_cf`

Inputs
------

*None*

Outputs
-------

- ``cutouts/{cutout}``: weather data from either the `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_
  reanalysis weather dataset or `SARAH-2 <https://wui.cmsaf.eu/safira/action/viewProduktSearch>`_
  satellite-based historic weather data with the following structure:

**ERA5 cutout:**

    ===================  ==========  ==========  =========================================================
    Field                Dimensions  Unit        Description
    ===================  ==========  ==========  =========================================================
    pressure             time, y, x  Pa          Surface pressure
    -------------------  ----------  ----------  ---------------------------------------------------------
    temperature          time, y, x  K           Air temperature 2 meters above the surface.
    -------------------  ----------  ----------  ---------------------------------------------------------
    soil temperature     time, y, x  K           Soil temperature between 1 meters and 3 meters
                                                 depth (layer 4).
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_toa           time, y, x  Wm**-2      Top of Earth's atmosphere TOA incident solar radiation
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_direct        time, y, x  Wm**-2      Total sky direct solar radiation at surface
    -------------------  ----------  ----------  ---------------------------------------------------------
    runoff               time, y, x  m           `Runoff <https://en.wikipedia.org/wiki/Surface_runoff>`_
                                                 (volume per area)
    -------------------  ----------  ----------  ---------------------------------------------------------
    roughness            y, x        m           Forecast surface roughness
                                                 (`roughness length <https://en.wikipedia.org/wiki/Roughness_length>`_)
    -------------------  ----------  ----------  ---------------------------------------------------------
    height               y, x        m           Surface elevation above sea level
    -------------------  ----------  ----------  ---------------------------------------------------------
    albedo               time, y, x  --          `Albedo <https://en.wikipedia.org/wiki/Albedo>`_
                                                 measure of diffuse reflection of solar radiation.
                                                 Calculated from relation between surface solar radiation
                                                 downwards (Jm**-2) and surface net solar radiation
                                                 (Jm**-2). Takes values between 0 and 1.
    -------------------  ----------  ----------  ---------------------------------------------------------
    influx_diffuse       time, y, x  Wm**-2      Diffuse solar radiation at surface.
                                                 Surface solar radiation downwards minus
                                                 direct solar radiation.
    -------------------  ----------  ----------  ---------------------------------------------------------
    wnd100m              time, y, x  ms**-1      Wind speeds at 100 meters (regardless of direction)
    ===================  ==========  ==========  =========================================================

    .. image:: ../img/era5.png
        :scale: 40 %

A **SARAH-2 cutout** can be used to amend the fields ``temperature``, ``influx_toa``, ``influx_direct``, ``albedo``,
``influx_diffuse`` of ERA5 using satellite-based radiation observations.

    .. image:: ../img/sarah.png
        :scale: 40 %

Description
-----------

"""

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging
import numpy as np
import pandas as pd
import geopandas as gpd

import os
import atlite

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_cutout', cutout='europe-2013-era5')
    configure_logging(snakemake)

    cutout_params = snakemake.config['atlite']['cutouts'][snakemake.wildcards.cutout]
    for p in ('xs', 'ys', 'years', 'months'):
        if p in cutout_params:
            cutout_params[p] = slice(*cutout_params[p])

    cutout = atlite.Cutout(snakemake.wildcards.cutout,
                        cutout_dir=os.path.dirname(snakemake.output[0]),
                        **cutout_params)

    # Check if all spatial and temporal bounds of the network are included in
    # the cutout
    cutout_bounds = np.array(cutout.extent)[[0,2,1,3]]
    onshore_bounds = gpd.read_file(snakemake.input.regions_onshore).total_bounds
    offshore_bounds = gpd.read_file(snakemake.input.regions_offshore).total_bounds

    assert all(onshore_bounds[:2] >= cutout_bounds[:2])
    assert all(onshore_bounds[2:] <= cutout_bounds[2:])

    # assert all(offshore_bounds[:2] >= cutout_bounds[:2])
    # assert all(offshore_bounds[2:] <= cutout_bounds[2:])

    snapshots = pd.date_range(freq='h', **snakemake.config['snapshots'])
    assert all(snapshots.isin(cutout.coords['time'].data))

    cutout.prepare(nprocesses=snakemake.config['atlite'].get('nprocesses', 4))
