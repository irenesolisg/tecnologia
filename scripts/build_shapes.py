# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Creates GIS shape files of the countries, exclusive economic zones and `NUTS3 <https://en.wikipedia.org/wiki/Nomenclature_of_Territorial_Units_for_Statistics>`_ areas.

Relevant Settings
-----------------

.. code:: yaml

    countries:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`

Inputs
------

- ``data/bundle/naturalearth/ne_10m_admin_0_countries.shp``: World country shapes

    .. image:: ../img/countries.png
        :scale: 33 %

- ``data/bundle/eez/World_EEZ_v8_2014.shp``: World `exclusive economic zones <https://en.wikipedia.org/wiki/Exclusive_economic_zone>`_ (EEZ)

    .. image:: ../img/eez.png
        :scale: 33 %

- ``data/bundle/NUTS_2013_60M_SH/data/NUTS_RG_60M_2013.shp``: Europe NUTS3 regions

    .. image:: ../img/nuts3.png
        :scale: 33 %

- ``data/bundle/nama_10r_3popgdp.tsv.gz``: Average annual population by NUTS3 region (`eurostat <http://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=nama_10r_3popgdp&lang=en>`__)
- ``data/bundle/nama_10r_3gdp.tsv.gz``: Gross domestic product (GDP) by NUTS 3 regions (`eurostat <http://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=nama_10r_3gdp&lang=en>`__)
- ``data/bundle/ch_cantons.csv``: Mapping between Swiss Cantons and NUTS3 regions
- ``data/bundle/je-e-21.03.02.xls``: Population and GDP data per Canton (`BFS - Swiss Federal Statistical Office <https://www.bfs.admin.ch/bfs/en/home/news/whats-new.assetdetail.7786557.html>`_ )

Outputs
-------

- ``resources/country_shapes.geojson``: country shapes out of country selection

    .. image:: ../img/country_shapes.png
        :scale: 33 %

- ``resources/offshore_shapes.geojson``: EEZ shapes out of country selection

    .. image:: ../img/offshore_shapes.png
        :scale: 33 %

- ``resources/europe_shape.geojson``: Shape of Europe including countries and EEZ

    .. image:: ../img/europe_shape.png
        :scale: 33 %

- ``resources/nuts3_shapes.geojson``: NUTS3 shapes out of country selection including population and GDP data.

    .. image:: ../img/nuts3_shapes.png
        :scale: 33 %

Description
-----------

"""

import logging
from _helpers import configure_logging

import os
import numpy as np
from operator import attrgetter
from functools import reduce
from itertools import takewhile

import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import pycountry as pyc

logger = logging.getLogger(__name__)


def _get_country(target, **keys):
    assert len(keys) == 1
    try:
        return getattr(pyc.countries.get(**keys), target)
    except (KeyError, AttributeError):
        return np.nan


def _simplify_polys(polys, minarea=0.1, tolerance=0.01, filterremote=True):
    if isinstance(polys, MultiPolygon):
        polys = sorted(polys.geoms, key=attrgetter('area'), reverse=True)
        mainpoly = polys[0]
        mainlength = np.sqrt(mainpoly.area/(2.*np.pi))
        if mainpoly.area > minarea:
            polys = MultiPolygon([p
                                  for p in takewhile(lambda p: p.area > minarea, polys)
                                  if not filterremote or (mainpoly.distance(p) < mainlength)])
        else:
            polys = mainpoly

    polys = polys.simplify(tolerance=tolerance)
    
    if not polys.is_valid:
        polys = make_valid(polys)

    return polys


def countries(naturalearth, country_list):
    if 'RS' in country_list: country_list.append('XK')

    df = gpd.read_file(naturalearth)

    # Names are a hassle in naturalearth, try several fields
    fieldnames = (df[x].where(lambda s: s!='-99') for x in ('ISO_A2', 'WB_A2', 'ADM0_A3'))
    df['name'] = reduce(lambda x,y: x.fillna(y), fieldnames, next(fieldnames)).str[0:2]

    df = df.loc[df.name.isin(country_list) & ((df['scalerank'] == 0) | (df['scalerank'] == 5))]
    s = df.set_index('name')['geometry'].map(_simplify_polys)
    if 'RS' in country_list: s['RS'] = s['RS'].union(s.pop('XK'))
    s["RS"] = Polygon(s["RS"].exterior)

    return s


def eez(eez, country_list):
    df = gpd.read_file(eez)
    iso3_list = [_get_country('alpha_3', alpha_2=c) for c in country_list]
    df = df.query("ISO_TER1 in @iso3_list and POL_TYPE == '200NM'")
    df['name'] = df['ISO_TER1'].map(lambda c: _get_country('alpha_2', alpha_3=c))
    s = df.set_index('name').geometry.map(lambda s: _simplify_polys(s, filterremote=False))
    s.index.name = "name"
    return s


def country_cover(country_shapes, eez_shapes=None):
    shapes = list(country_shapes)
    if eez_shapes is not None:
        shapes += list(eez_shapes)

    europe_shape = unary_union(shapes)
    if isinstance(europe_shape, MultiPolygon):
        europe_shape = max(europe_shape, key=attrgetter('area'))
    return Polygon(shell=europe_shape.exterior)


def nuts3(country_shapes, nuts3, nuts3pop, nuts3gdp, ch_cantons, ch_popgdp):
    df = gpd.read_file(nuts3)
    df = df.loc[df['STAT_LEVL_'] == 3]
    df['geometry'] = df['geometry'].map(_simplify_polys)
    df = df.rename(columns={'NUTS_ID': 'id'})[['id', 'geometry']].set_index('id')

    pop = pd.read_table(nuts3pop, na_values=[':'], delimiter=' ?\t', engine='python')
    pop = (pop
           .set_index(pd.MultiIndex.from_tuples(pop.pop('unit,geo\\time').str.split(','))).loc['THS']
           .applymap(lambda x: pd.to_numeric(x, errors='coerce'))
           .fillna(method='bfill', axis=1))['2014']

    gdp = pd.read_table(nuts3gdp, na_values=[':'], delimiter=' ?\t', engine='python')
    gdp = (gdp
           .set_index(pd.MultiIndex.from_tuples(gdp.pop('unit,geo\\time').str.split(','))).loc['EUR_HAB']
           .applymap(lambda x: pd.to_numeric(x, errors='coerce'))
           .fillna(method='bfill', axis=1))['2014']

    cantons = pd.read_csv(ch_cantons)
    cantons = cantons.set_index(cantons['HASC'].str[3:])['NUTS']
    cantons = cantons.str.pad(5, side='right', fillchar='0')

    swiss = pd.read_excel(ch_popgdp, skiprows=3, index_col=0)
    swiss.columns = swiss.columns.to_series().map(cantons)

    swiss_pop = pd.to_numeric(swiss.loc['Residents in 1000', 'CH040':])
    pop = pd.concat([pop, swiss_pop])
    swiss_gdp = pd.to_numeric(swiss.loc['Gross domestic product per capita in Swiss francs', 'CH040':])
    gdp = pd.concat([gdp, swiss_gdp])

    df = df.join(pd.DataFrame(dict(pop=pop, gdp=gdp)))

    df['country'] = df.index.to_series().str[:2].replace(dict(UK='GB', EL='GR'))

    excludenuts = pd.Index(('FRA10', 'FRA20', 'FRA30', 'FRA40', 'FRA50',
                            'PT200', 'PT300',
                            'ES707', 'ES703', 'ES704','ES705', 'ES706', 'ES708', 'ES709',
                            'FI2', 'FR9'))
    excludecountry = pd.Index(('MT', 'TR', 'LI', 'IS', 'CY', 'KV'))

    df = df.loc[df.index.difference(excludenuts)]
    df = df.loc[~df.country.isin(excludecountry)]

    manual = gpd.GeoDataFrame(
        [['BA1', 'BA', 3871.],
         ['RS1', 'RS', 7210.],
         ['AL1', 'AL', 2893.]],
        columns=['NUTS_ID', 'country', 'pop']
    ).set_index('NUTS_ID')
    manual['geometry'] = manual['country'].map(country_shapes)
    manual = manual.dropna()

    df = pd.concat([df, manual], sort=False)

    df.loc['ME000', 'pop'] = 650.

    return df


def save_to_geojson(df, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(dict(geometry=df))
    df = df.reset_index()
    schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
    df.to_file(fn, driver='GeoJSON', schema=schema)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_shapes')
    configure_logging(snakemake)

    country_shapes = countries(snakemake.input.naturalearth, snakemake.config['countries'])
    save_to_geojson(country_shapes, snakemake.output.country_shapes)

    offshore_shapes = eez(snakemake.input.eez, snakemake.config['countries'])
    save_to_geojson(offshore_shapes, snakemake.output.offshore_shapes)

    europe_shape = country_cover(country_shapes, offshore_shapes)
    save_to_geojson(gpd.GeoSeries(europe_shape), snakemake.output.europe_shape)

    nuts3_shapes = nuts3(country_shapes, snakemake.input.nuts3, snakemake.input.nuts3pop,
                         snakemake.input.nuts3gdp, snakemake.input.ch_cantons, snakemake.input.ch_popgdp)

    save_to_geojson(nuts3_shapes, snakemake.output.nuts3_shapes)
