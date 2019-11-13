# coding: utf-8
"""
Adds electrical generators and storage units to a base network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        USD2013_to_EUR2013:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        conventional_carriers:
        co2limit:
        extendable_carriers:
            Generator:
            StorageUnit:
        estimate_renewable_capacities_from_capacity_stats:

    load:
        scaling_factor:

    renewable: (keys)
        hydro:
            carriers:
            hydro_max_hours:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------

- ``data/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/bundle/hydro_capacities.csv``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.

    .. image:: ../img/hydrocapacities.png
        :scale: 34 %

- ``data/geth2015_hydro_capacities.csv``: alternative to capacities above; NOT CURRENTLY USED!
- ``data/bundle/time_series_60min_singleindex_filtered.csv``: Hourly per-country load profiles since 2010 from the `ENTSO-E statistical database <https://www.entsoe.eu/data/power-stats/hourly_load/>`_

    .. image:: ../img/load-box.png
        :scale: 33 %

    .. image:: ../img/load-ts.png
        :scale: 33 %

- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/nuts3_shapes.geojson``: confer :ref:`shapes`
- ``resources/powerplants.csv``: confer :ref:`powerplants`
- ``resources/profile_{}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec.nc``:

    .. image:: ../img/elec.png
            :scale: 33 %

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today's transmission topology and transfer capacities (optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today's thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today's load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` and ``storage_units`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today's locational, hourly wind and solar capacity factors (but **no** current capacities),
- long-term hydrogen and short-term battery storage units (if listed in the config setting ``electricity: extendable_carriers``), and
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

import logging
logger = logging.getLogger(__name__)

import pandas as pd
idx = pd.IndexSlice

import numpy as np
import xarray as xr

import geopandas as gpd

from vresutils.costdata import annuity
from vresutils.load import timeseries_opsd
from vresutils import transfer as vtransfer

import pypsa
import powerplantmatching as ppm


def normed(s): return s/s.sum()

def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty: return

    emissions_cols = costs.columns.to_series()\
                           .loc[lambda s: s.str.endswith('_emissions')].values
    suptechs = missing_carriers.str.split('-').str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, 'Carrier')

def load_costs(Nyears=1., tech_costs=None, config=None, elec_config=None):
    if tech_costs is None:
        tech_costs = snakemake.input.tech_costs

    if config is None:
        config = snakemake.config['costs']

    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=list(range(3))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"),"value"] *= config['USD2013_to_EUR2013']

    costs = (costs.loc[idx[:,config['year'],:], "value"]
             .unstack(level=2).groupby("technology").sum(min_count=1))

    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : config['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25})

    costs["capital_cost"] = ((annuity(costs["lifetime"], costs["discount rate"]) +
                             costs["FOM"]/100.) *
                             costs["investment"] * Nyears)

    costs.at['OCGT', 'fuel'] = costs.at['gas', 'fuel']
    costs.at['CCGT', 'fuel'] = costs.at['gas', 'fuel']

    costs['marginal_cost'] = costs['VOM'] + costs['fuel'] / costs['efficiency']

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at['OCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']
    costs.at['CCGT', 'co2_emissions'] = costs.at['gas', 'co2_emissions']

    costs.at['solar', 'capital_cost'] = 0.5*(costs.at['solar-rooftop', 'capital_cost'] +
                                             costs.at['solar-utility', 'capital_cost'])

    def costs_for_storage(store, link1, link2=None, max_hours=1.):
        capital_cost = link1['capital_cost'] + max_hours * store['capital_cost']
        efficiency = link1['efficiency']**0.5
        if link2 is not None:
            capital_cost += link2['capital_cost']
            efficiency *= link2['efficiency']**0.5
        return pd.Series(dict(capital_cost=capital_cost,
                              marginal_cost=0.,
                              efficiency=efficiency,
                              co2_emissions=0.))

    if elec_config is None:
        elec_config = snakemake.config['electricity']
    max_hours = elec_config['max_hours']
    costs.loc["battery"] = \
        costs_for_storage(costs.loc["battery storage"], costs.loc["battery inverter"],
                          max_hours=max_hours['battery'])
    costs.loc["H2"] = \
        costs_for_storage(costs.loc["hydrogen storage"], costs.loc["fuel cell"],
                          costs.loc["electrolysis"], max_hours=max_hours['H2'])

    for attr in ('marginal_cost', 'capital_cost'):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs

def load_powerplants(ppl_fn=None):
    if ppl_fn is None:
        ppl_fn = snakemake.input.powerplants
    carrier_dict = {'ocgt': 'OCGT', 'ccgt': 'CCGT', 'bioenergy':'biomass',
                    'ccgt, thermal': 'CCGT', 'hard coal': 'coal'}
    return (pd.read_csv(ppl_fn, index_col=0, dtype={'bus': 'str'})
            .powerplant.to_pypsa_names()
            .rename(columns=str.lower).drop(columns=['efficiency'])
            .replace({'carrier': carrier_dict}))


# =============================================================================
# Attach components
# =============================================================================

# ### Load

def attach_load(n):
    substation_lv_i = n.buses.index[n.buses['substation_lv']]
    regions = (gpd.read_file(snakemake.input.regions).set_index('name')
               .reindex(substation_lv_i))
    opsd_load = (timeseries_opsd(slice(*n.snapshots[[0,-1]].year.astype(str)),
                                 snakemake.input.opsd_load) *
                 snakemake.config.get('load', {}).get('scaling_factor', 1.0))

    # Convert to naive UTC (has to be explicit since pandas 0.24)
    opsd_load.index = opsd_load.index.tz_localize(None)

    nuts3 = gpd.read_file(snakemake.input.nuts3_shapes).set_index('index')

    def normed(x): return x.divide(x.sum())

    def upsample(cntry, group):
        l = opsd_load[cntry]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            nuts3_cntry = nuts3.loc[nuts3.country == cntry]
            transfer = vtransfer.Shapes2Shapes(group, nuts3_cntry.geometry,
                                               normed=False).T.tocsr()
            gdp_n = pd.Series(transfer.dot(nuts3_cntry['gdp'].fillna(1.).values),
                              index=group.index)
            pop_n = pd.Series(transfer.dot(nuts3_cntry['pop'].fillna(1.).values),
                              index=group.index)

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to continent load data (refer to vresutils.load._upsampling_weights)
            factors = normed(0.6 * normed(gdp_n) + 0.4 * normed(pop_n))
            return pd.DataFrame(factors.values * l.values[:,np.newaxis],
                                index=l.index, columns=factors.index)

    load = pd.concat([upsample(cntry, group) for cntry, group
                      in regions.geometry.groupby(regions.country)], axis=1)

    n.madd("Load", substation_lv_i, bus=substation_lv_i, p_set=load)

### Set line costs

def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
    n.lines['capital_cost'] = (n.lines['length'] * length_factor *
                               costs.at['HVAC overhead', 'capital_cost'])

    if n.links.empty: return

    dc_b = n.links.carrier == 'DC'
    if simple_hvdc_costs:
        costs = (n.links.loc[dc_b, 'length'] * length_factor *
                 costs.at['HVDC overhead', 'capital_cost'])
    else:
        costs = (n.links.loc[dc_b, 'length'] * length_factor *
                ((1. - n.links.loc[dc_b, 'underwater_fraction']) *
                costs.at['HVDC overhead', 'capital_cost'] +
                n.links.loc[dc_b, 'underwater_fraction'] *
                costs.at['HVDC submarine', 'capital_cost']) +
                costs.at['HVDC inverter pair', 'capital_cost'])
    n.links.loc[dc_b, 'capital_cost'] = costs
# ### Generators

def attach_wind_and_solar(n, costs):
    for tech in snakemake.config['renewable']:
        if tech == 'hydro': continue

        n.add("Carrier", name=tech)
        with xr.open_dataset(getattr(snakemake.input, 'profile_' + tech)) as ds:
            if ds.indexes['bus'].empty: continue

            suptech = tech.split('-', 2)[0]
            if suptech == 'offwind':
                underwater_fraction = ds['underwater_fraction'].to_pandas()
                connection_cost = (snakemake.config['lines']['length_factor'] *
                                   ds['average_distance'].to_pandas() *
                                   (underwater_fraction *
                                    costs.at[tech + '-connection-submarine', 'capital_cost'] +
                                    (1. - underwater_fraction) *
                                    costs.at[tech + '-connection-underground', 'capital_cost']))
                capital_cost = (costs.at['offwind', 'capital_cost'] +
                                costs.at[tech + '-station', 'capital_cost'] +
                                connection_cost)
                logger.info("Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}"
                            .format(connection_cost.min(), connection_cost.max(), tech))
            elif suptech == 'onwind':
                capital_cost = (costs.at['onwind', 'capital_cost'] +
                                costs.at['onwind-landcosts', 'capital_cost'])
            else:
                capital_cost = costs.at[tech, 'capital_cost']

            n.madd("Generator", ds.indexes['bus'], ' ' + tech,
                   bus=ds.indexes['bus'],
                   carrier=tech,
                   p_nom_extendable=True,
                   p_nom_max=ds['p_nom_max'].to_pandas(),
                   weight=ds['weight'].to_pandas(),
                   marginal_cost=costs.at[suptech, 'marginal_cost'],
                   capital_cost=capital_cost,
                   efficiency=costs.at[suptech, 'efficiency'],
                   p_max_pu=ds['profile'].transpose('time', 'bus').to_pandas())


# # Generators


def attach_conventional_generators(n, costs, ppl):
    carriers = snakemake.config['electricity']['conventional_carriers']
    _add_missing_carriers_from_costs(n, costs, carriers)
    ppl = (ppl.query('carrier in @carriers').join(costs, on='carrier')
           .rename(index=lambda s: 'C' + str(s)))

    logger.info('Adding {} generators with capacities\n{}'
                .format(len(ppl), ppl.groupby('carrier').p_nom.sum()))
    n.madd("Generator", ppl.index,
           carrier=ppl.carrier,
           bus=ppl.bus,
           p_nom=ppl.p_nom,
           efficiency=ppl.efficiency,
           marginal_cost=ppl.marginal_cost,
           capital_cost=0)
    logger.warning(f'Capital costs for conventional generators put to 0 EUR/MW.')


def attach_hydro(n, costs, ppl):
    c = snakemake.config['renewable']['hydro']
    carriers = c.get('carriers', ['ror', 'PHS', 'hydro'])

    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = ppl.query('carrier == "hydro"').reset_index(drop=True)\
             .rename(index=lambda s: str(s) + ' hydro')
    ror = ppl.query('technology == "Run-Of-River"')
    phs = ppl.query('technology == "Pumped Storage"')
    hydro = ppl.query('technology == "Reservoir"')

    country = ppl['bus'].map(n.buses.country).rename("country")

    inflow_idx = ror.index | hydro.index
    if not inflow_idx.empty:
        dist_key = ppl.loc[inflow_idx, 'p_nom'].groupby(country).transform(normed)

        with xr.open_dataarray(snakemake.input.profile_hydro) as inflow:
            inflow_countries = pd.Index(country[inflow_idx])
            missing_c = (inflow_countries.unique()
                         .difference(inflow.indexes['countries']))
            assert missing_c.empty, (f"'{snakemake.input.profile_hydro}' is missing "
                f"inflow time-series for at least one country: {', '.join(missing_c)}")

            inflow_t = (inflow.sel(countries=inflow_countries)
                        .rename({'countries': 'name'})
                        .assign_coords(name=inflow_idx)
                        .transpose('time', 'name')
                        .to_pandas()
                        .multiply(dist_key, axis=1))

    if 'ror' in carriers and not ror.empty:
        n.madd("Generator", ror.index,
               carrier='ror',
               bus=ror['bus'],
               p_nom=ror['p_nom'],
               efficiency=costs.at['ror', 'efficiency'],
               capital_cost=costs.at['ror', 'capital_cost'],
               weight=ror['p_nom'],
               p_max_pu=(inflow_t[ror.index]
                         .divide(ror['p_nom'], axis=1)
                         .where(lambda df: df<=1., other=1.)))

    if 'PHS' in carriers and not phs.empty:
        # fill missing max hours to config value and assume no natural inflow
        # due to lack of data
        phs = phs.replace({'max_hours': {0: c['PHS_max_hours']}})
        n.madd('StorageUnit', phs.index,
               carrier='PHS',
               bus=phs['bus'],
               p_nom=phs['p_nom'],
               capital_cost=costs.at['PHS', 'capital_cost'],
               max_hours=phs['max_hours'],
               efficiency_store=np.sqrt(costs.at['PHS','efficiency']),
               efficiency_dispatch=np.sqrt(costs.at['PHS','efficiency']),
               cyclic_state_of_charge=True)

    if 'hydro' in carriers and not hydro.empty:
        hydro_max_hours = c.get('hydro_max_hours')
        hydro_stats = pd.read_csv(snakemake.input.hydro_capacities,
                                   comment="#", na_values='-', index_col=0)
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval('p_nom * max_hours').groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query('max_hours == 0').index

        if hydro_max_hours == 'energy_capacity_totals_by_country':
            # watch out some p_nom values like IE's are totally underrepresented
            max_hours_country = e_missing / \
                                hydro.loc[missing_mh_i].groupby('country').p_nom.sum()

        elif hydro_max_hours == 'estimate_by_large_installations':
            max_hours_country = hydro_stats['E_store[TWh]'] * 1e3 / \
                                hydro_stats['p_nom_discharge[GW]']

        missing_countries = (pd.Index(hydro['country'].unique())
                             .difference(max_hours_country.dropna().index))
        if not missing_countries.empty:
            logger.warning("Assuming max_hours=6 for hydro reservoirs in the countries: {}"
                           .format(", ".join(missing_countries)))
        hydro_max_hours = hydro.max_hours.where(hydro.max_hours > 0,
                                hydro.country.map(max_hours_country)).fillna(6)


        n.madd('StorageUnit', hydro.index, carrier='hydro',
               bus=hydro['bus'],
               p_nom=hydro['p_nom'],
               max_hours=hydro_max_hours,
               capital_cost=(costs.at['hydro', 'capital_cost']
                             if c.get('hydro_capital_cost') else 0.),
               marginal_cost=costs.at['hydro', 'marginal_cost'],
               p_max_pu=1.,  # dispatch
               p_min_pu=0.,  # store
               efficiency_dispatch=costs.at['hydro', 'efficiency'],
               efficiency_store=0.,
               cyclic_state_of_charge=True,
               inflow=inflow_t.loc[:, hydro.index])


def attach_extendable_generators(n, costs, ppl):
    elec_opts = snakemake.config['electricity']
    carriers = pd.Index(elec_opts['extendable_carriers']['Generator'])

    _add_missing_carriers_from_costs(n, costs, carriers)

    for tech in carriers:
        suptech = tech.split('-')[0]

        if suptech == 'OCGT':
            ocgt = ppl.query("carrier in ['OCGT', 'CCGT']").groupby('bus', as_index=False).first()
            n.madd('Generator', ocgt.index,
                   suffix=' OCGT',
                   bus=ocgt['bus'],
                   carrier=tech,
                   p_nom_extendable=True,
                   p_nom=0.,
                   capital_cost=costs.at['OCGT', 'capital_cost'],
                   marginal_cost=costs.at['OCGT', 'marginal_cost'],
                   efficiency=costs.at['OCGT', 'efficiency'])

        elif suptech == 'CCGT':
            ccgt = ppl.query("carrier in ['OCGT', 'CCGT']").groupby('bus', as_index=False).first()
            n.madd('Generator', ccgt.index,
                   suffix=' CCGT',
                   bus=ccgt['bus'],
                   carrier=tech,
                   p_nom_extendable=True,
                   p_nom=0.,
                   capital_cost=costs.at['CCGT', 'capital_cost'],
                   marginal_cost=costs.at['CCGT', 'marginal_cost'],
                   efficiency=costs.at['CCGT', 'efficiency'])

        elif suptech == 'nuclear':
            nuclear = ppl.query("carrier == 'nuclear'").groupby('bus', as_index=False).first()
            n.madd('Generator', nuclear.index,
                suffix=' nuclear',
                bus=nuclear['bus'],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.,
                capital_cost=costs.at['nuclear', 'capital_cost'],
                marginal_cost=costs.at['nuclear', 'marginal_cost'],
                efficiency=costs.at['nuclear', 'efficiency'])

        else:
            raise NotImplementedError(f"Adding extendable generators for carrier "
                                      "'{tech}' is not implemented, yet. "
                                      "Only OCGT, CCGT and nuclear are allowed at the moment.")


def attach_storage(n, costs):
    elec_opts = snakemake.config['electricity']
    carriers = elec_opts['extendable_carriers']['StorageUnit']
    max_hours = elec_opts['max_hours']

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index[n.buses.substation_lv]

    for carrier in carriers:
        n.madd("StorageUnit", buses_i, ' ' + carrier,
               bus=buses_i,
               carrier=carrier,
               p_nom_extendable=True,
               capital_cost=costs.at[carrier, 'capital_cost'],
               marginal_cost=costs.at[carrier, 'marginal_cost'],
               efficiency_store=costs.at[carrier, 'efficiency'],
               efficiency_dispatch=costs.at[carrier, 'efficiency'],
               max_hours=max_hours[carrier],
               cyclic_state_of_charge=True)

    ## Implementing them separately will come later!
    ##
    # if 'H2' in carriers:
    #     h2_buses = n.madd("Bus", buses + " H2", carrier="H2")

    #     n.madd("Link", h2_buses + " Electrolysis",
    #            bus1=h2_buses,
    #            bus0=buses,
    #            p_nom_extendable=True,
    #            efficiency=costs.at["electrolysis", "efficiency"],
    #            capital_cost=costs.at["electrolysis", "capital_cost"])

    #     n.madd("Link", h2_buses + " Fuel Cell",
    #            bus0=h2_buses,
    #            bus1=buses,
    #            p_nom_extendable=True,
    #            efficiency=costs.at["fuel cell", "efficiency"],
    #            #NB: fixed cost is per MWel
    #            capital_cost=costs.at["fuel cell", "capital_cost"] * costs.at["fuel cell", "efficiency"])

    #     n.madd("Store", h2_buses,
    #            bus=h2_buses,
    #            e_nom_extendable=True,
    #            e_cyclic=True,
    #            capital_cost=costs.at["hydrogen storage", "capital_cost"])

    # if 'battery' in carriers:
    #     b_buses = n.madd("Bus", buses + " battery", carrier="battery")

    #     network.madd("Store", b_buses,
    #                  bus=b_buses,
    #                  e_cyclic=True,
    #                  e_nom_extendable=True,
    #                  capital_cost=costs.at['battery storage', 'capital_cost'])

    #     network.madd("Link", b_buses + " charger",
    #                  bus0=buses,
    #                  bus1=b_buses,
    #                  efficiency=costs.at['battery inverter', 'efficiency']**0.5,
    #                  capital_cost=costs.at['battery inverter', 'capital_cost'],
    #                  p_nom_extendable=True)

    #     network.madd("Link",
    #                  nodes + " battery discharger",
    #                  bus0=nodes + " battery",
    #                  bus1=nodes,
    #                  efficiency=costs.at['battery inverter','efficiency']**0.5,
    #                  marginal_cost=options['marginal_cost_storage'],
    #                  p_nom_extendable=True)

def estimate_renewable_capacities(n, tech_map=None):
    if tech_map is None:
        tech_map = (snakemake.config['electricity']
                    .get('estimate_renewable_capacities_from_capacity_stats', {}))

    if len(tech_map) == 0: return

    capacities = (ppm.data.Capacity_stats().powerplant.convert_country_to_alpha2()
                  [lambda df: df.Energy_Source_Level_2]
                  .set_index(['Fueltype', 'Country']).sort_index())

    countries = n.buses.country.unique()

    for ppm_fueltype, techs in tech_map.items():
        tech_capacities = capacities.loc[ppm_fueltype, 'Capacity']\
                                    .reindex(countries, fill_value=0.)
        tech_i = n.generators.query('carrier in @techs').index
        n.generators.loc[tech_i, 'p_nom'] = (
            (n.generators_t.p_max_pu[tech_i].mean() *
             n.generators.loc[tech_i, 'p_nom_max']) # maximal yearly generation
             .groupby(n.generators.bus.map(n.buses.country)) # for each country
             .transform(lambda s: normed(s) * tech_capacities.at[s.name])
             .where(lambda s: s>0.1, 0.))  # only capacities above 100kW

def add_co2limit(n, Nyears=1.):
    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=snakemake.config['electricity']['co2limit'] * Nyears)

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def add_nice_carrier_names(n):
    nice_names = pd.Series(snakemake.config['plotting']['nice_names'])
    n.carriers['nice_names'] = nice_names[n.carriers.index]


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict

        snakemake = MockSnakemake(output=['networks/elec.nc'])
        snakemake.input = snakemake.expand(
            Dict(base_network='networks/base.nc',
                 tech_costs='data/costs.csv',
                 regions="resources/regions_onshore.geojson",
                 powerplants="resources/powerplants.csv",
                 hydro_capacities='data/bundle/hydro_capacities.csv',
                 opsd_load='data/bundle/time_series_60min_singleindex_filtered.csv',
                 nuts3_shapes='resources/nuts3_shapes.geojson',
                 **{'profile_' + t: "resources/profile_" + t + ".nc"
                    for t in snakemake.config['renewable']})
        )

    logging.basicConfig(level=snakemake.config['logging_level'])

    n = pypsa.Network(snakemake.input.base_network)
    Nyears = n.snapshot_weightings.sum()/8760.

    costs = load_costs(Nyears)
    ppl = load_powerplants()

    attach_load(n)

    update_transmission_costs(n, costs)
    attach_conventional_generators(n, costs, ppl)

    attach_wind_and_solar(n, costs)
    if 'hydro' in snakemake.config['renewable']:
        attach_hydro(n, costs, ppl)
    attach_extendable_generators(n, costs, ppl)
    attach_storage(n, costs)

    estimate_renewable_capacities(n)
    add_nice_carrier_names(n)

    n.export_to_netcdf(snakemake.output[0])
