#!/usr/bin/env python

import os
import atlite
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from pypsa.geo import haversine
from vresutils.array import spdiag

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=snakemake.config['logging_level'])

config = snakemake.config['renewable'][snakemake.wildcards.technology]

time = pd.date_range(freq='m', **snakemake.config['snapshots'])
params = dict(years=slice(*time.year[[0, -1]]), months=slice(*time.month[[0, -1]]))

regions = gpd.read_file(snakemake.input.regions).set_index('name')
regions.index.name = 'bus'

cutout = atlite.Cutout(config['cutout'],
                       cutout_dir=os.path.dirname(snakemake.input.cutout),
                       **params)

# Potentials
potentials = xr.open_dataarray(snakemake.input.potentials)

# Indicatormatrix
indicatormatrix = cutout.indicatormatrix(regions.geometry)

resource = config['resource']
func = getattr(cutout, resource.pop('method'))
correction_factor = config.get('correction_factor', 1.)
if correction_factor != 1.:
    logger.warning('correction_factor is set as {}'.format(correction_factor))
capacity_factor = correction_factor * func(capacity_factor=True, **resource)
layout = capacity_factor * potentials

profile, capacities = func(matrix=indicatormatrix, index=regions.index,
                           layout=layout, per_unit=True, return_capacity=True,
                           **resource)

relativepotentials = (potentials / layout).stack(spatial=('y', 'x')).values
p_nom_max = xr.DataArray([np.nanmin(relativepotentials[row.nonzero()[1]])
                          if row.getnnz() > 0 else 0
                          for row in indicatormatrix.tocsr()],
                         [capacities.coords['bus']]) * capacities

# Determine weighted average distance to substation
matrix_weighted = indicatormatrix * spdiag(layout.stack(spatial=('y', 'x')))
cell_coords = cutout.grid_coordinates()

average_distance = []
for i, bus in enumerate(regions.index):
    row = matrix_weighted[i]
    distances = haversine(regions.loc[bus, ['x', 'y']], cell_coords[row.indices])[0]
    average_distance.append((distances * (row.data / row.data.sum())).sum())
average_distance = xr.DataArray(average_distance, [regions.index.rename("bus")])

ds = xr.merge([(correction_factor * profile).rename('profile'),
               capacities.rename('weight'),
               p_nom_max.rename('p_nom_max'),
               layout.rename('potential'),
               average_distance.rename('average_distance')])
(ds.sel(bus=ds['profile'].mean('time') > config.get('min_p_max_pu', 0.))
 .to_netcdf(snakemake.output.profile))
