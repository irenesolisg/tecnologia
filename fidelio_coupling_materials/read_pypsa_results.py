# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:14:43 2023

@author: alice
"""

#%% Import the libraries needed

import pypsa

import os

import pycountry
import pandas as pd



def get_country_name(alpha_2_code):
    try:
        country = pycountry.countries.get(alpha_2=alpha_2_code)
        return country.name
    except (ValueError, AttributeError, LookupError):
        return 'Unknown'  # Return 'Unknown' if the code is not valid or the country name cannot be found

# Figures out the absolute path for you in case your working directory moves around.
my_path = os.getcwd()
parent_directory = os.path.dirname(my_path)
#os.chdir(parent_directory)

#%% Get energy prices iteratively for the 3 selected years
years = ['2030','2040','2050']
changes = pd.DataFrame(columns = years)

for year in years:
    network = pypsa.Network(f"./results/ff55/postnetworks/elec_s_39_lvopt__Co2P-ff55_{year}.nc")
    network_baseline = pypsa.Network(f"./results/baseline/postnetworks/elec_s_39_lvopt__Co2P-ff55_{year}.nc")
    
    mprice = network.buses_t.marginal_price
    elec_prices = mprice[[col for col in mprice.columns if "low voltage" in col]]
    elec_prices = elec_prices.rename(columns=lambda x: x.replace('low voltage', '').strip())
    
    
    mprice_default = network_baseline.buses_t.marginal_price
    elec_prices_default = mprice_default[[col for col in mprice_default.columns if "low voltage" in col]]
    elec_prices_default = elec_prices_default.rename(columns=lambda x: x.replace('low voltage', '').strip())
    
    # Get electricity loads per hours
    
    loads = network.loads_t.p
    elec_loads = loads[[col for col in loads.columns if col.endswith('0')]]
    
    loads_default = network_baseline.loads_t.p
    elec_loads_default = loads_default[[col for col in loads_default.columns if col.endswith('0')]]
    
    # Get total prices per hour
    total_elec_prices = elec_prices*elec_loads
    total_elec_prices_default = elec_prices_default*elec_loads_default
    
    # Get the weighted average over electricity consumption of the electricity price
    weighted_average = total_elec_prices.sum()/elec_loads.sum()
    weighted_average_default = total_elec_prices_default.sum()/elec_loads_default.sum()
    
    def average_same_country(weighted_average):
        for i in range(len(weighted_average.index.str[:2].unique())-1):
        
            if weighted_average.index[i][0:2] == weighted_average.index[i+1][0:2]:
                weighted_average[weighted_average.index[i]] = (weighted_average[weighted_average.index[i]] * elec_loads.sum()[i] + weighted_average[weighted_average.index[i+1]] * elec_loads.sum()[i+1]) /(elec_loads.sum()[i]+elec_loads.sum()[i+1])
                weighted_average = weighted_average.drop(weighted_average.index[i+1])
        
        return weighted_average
    
    weighted_average = average_same_country(weighted_average)
    weighted_average_default = average_same_country(weighted_average_default)
    
    changes[year] = ((weighted_average-weighted_average_default)/weighted_average_default)*100
    
# Extracting the first two characters of the index to specify the country name
# Assuming your DataFrame is named df
changes = changes.reset_index().rename(columns={'Bus': 'Country'})
changes['Country'] = changes['Country'].str[:2].apply(get_country_name)
    
changes.to_csv(f'./fidelio_coupling_materials/perc_changes_elec_prices.csv', index=False)


# %% Retrieve share RES per year and scenario

years = ['2030','2040', '2050'] #Need to add 2020
scenarios = ['baseline','ff55']

database = pd.DataFrame()

for scenario in scenarios:
    for i in years:
        
        if i == '2020':
            
            file_name = f"elec_s_39_lvopt__Co2P-ff55_{i}"
            
        else:

            file_name = f"elec_s_39_lvopt__Co2P-ff55_{i}"

        network = pypsa.Network(f"./results/{scenario}/postnetworks/" + file_name + ".nc")
        
        # Read the variables in the newtwork
        variables = dir(network)
        
        timestep = network.snapshot_weightings.iloc[0,0] # HOURS TO MULTIPLY FOR THE ANNUAL VALUES

        
        year = file_name.split('_')[-1]
        model = 'ESOPUS'
        region_default = 'Europe'
        #scenario = [string for string in file_name.split('_') if string.startswith('Co2')][0][len("Co2L"):].lstrip('-').split("-")[0]
        #scenario = 'ff55'
        #twh_to_ej = 3.6/1000
    
        
        countries = ['Albania', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria',
               'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
               'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia',
               'Lithuania', 'Luxembourg', 'Montenegro', 'Netherlands', 'North Macedonia',
               'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia',
               'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']
        
        # FINAL ENERGY 
        
        annual_demand = network.loads_t.p.sum()*timestep*1e-6  # TWh

        alinks = network.links
        filtered_links1 = alinks[alinks.filter(like='bus1').apply(lambda col: col.str.contains('heat')).any(axis=1)].index
        filtered_links2 = alinks[alinks.filter(like='bus2').apply(lambda col: col.str.contains('heat')).any(axis=1)].index
        filtered_links3 = alinks[alinks.filter(like='bus3').apply(lambda col: col.str.contains('heat')).any(axis=1)].index
        
        conventional1 = network.links_t.p1.loc[:,network.links_t.p1.columns.isin(filtered_links1)].sum()*timestep*1e-6 # TWh
        conventional2 = network.links_t.p2.loc[:,network.links_t.p2.columns.isin(filtered_links2)].sum()*timestep*1e-6 # TWh
        conventional3 = network.links_t.p3.loc[:,network.links_t.p2.columns.isin(filtered_links3)].sum()*timestep*1e-6 # TWh
        
        # Split the series based on the sign of values, positive for HEAT DEMAND, negative for HEAT PRODUCTION
        demand_conventional1 = conventional1[conventional1 > 0]
        demand_conventional2 = conventional2[conventional2 > 0]
        demand_conventional3 = conventional3[conventional3 > 0]  
        
        annual_demand = pd.concat([annual_demand,demand_conventional3])
        
        # Rearrange annual generation dataframe
        annual_demand = (pd.concat([annual_demand.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),
                    annual_demand.index.str.extract(r'([A-Z][A-Z])', expand=True),
                    annual_demand.index.str.extract(r'[A-Z]?[A-Z]?\d? ?\d? ?(.+)', expand=True),
                    annual_demand.reset_index()],ignore_index=True, axis=1).drop(3, axis=1)
                    .rename(columns={0: 'Node', 1: 'Country', 2: 'Demand type', 4: 'Demand [TWh/yr]'}))
      
        # Substituting nan and calling countries with a nice name
        annual_demand['Country'] = annual_demand['Country'].apply(get_country_name)
        annual_demand['Country'] = annual_demand['Country'].replace('Unknown', 'Europe') # In this case the unknow countries are aggregated for Europe
        
        annual_demand = annual_demand.groupby(['Country','Demand type'])['Demand [TWh/yr]'].sum()
        annual_demand = annual_demand.reset_index(drop=False)
        
        # Assume that unspecified nodes refer to EU
        annual_demand = annual_demand[annual_demand['Country'] != 'EU']
        
        # Assume that unspecified demand type refer to electricity of residential and services sectors
        annual_demand['Demand type'] = annual_demand['Demand type'].replace('0', 'residential and services electricity')
        
        # Assign energy carrier to generation type
        annual_demand.insert(2, 'Carrier', ['electricity' if ('electricity' in x) or ('EV' in x) else
                                            ('heat' if ('heat' in x) or ('DAC' in x) else
                                             ('biomass' if 'biomass' in x else
                                              ('hydrogen' if ('H2' in x) or ('fuel cell' in x) else
                                               ('gas' if 'gas' in x else
                                                ('oil' if ('oil' in x and 'emissions' not in x) else
                                                 ('methanol' if ('methanol' in x and 'emissions' not in x) else
                                                  ('kerosene' if 'kerosene' in x else
                                                   ('naphtha' if 'naphtha' in x else
                                                    ('CO2' if 'emissions' in x else
                                                     '')))))))))
                                            for x in annual_demand['Demand type']], True)
        
        annual_demand.insert(3, 'Sector', ['residential and services' if ('residential and services' in x) or ('urban central' in x) else
                                           ('services' if 'services' in x else
                                            ('residential' if 'residential' in x else
                                             ('industry' if ('industry' in x) or ('process' in x) else
                                              ('agriculture' if 'agriculture' in x else
                                               ('transportation' if ('land transport' in x) or ('shipping' in x) or ('aviation' in x) else
                                                ('industry, agriculture, and transportation' if 'oil emissions' in x else
                                                 ''))))))
                                           for x in annual_demand['Demand type']], True)
        
        
        # FINAL ENERGY
        
        annual_demand = annual_demand[~annual_demand['Carrier'].str.contains('CO2')]
        
        final_energy_europe = annual_demand['Demand [TWh/yr]'].sum()
        demand_sum_by_country = annual_demand.groupby('Country')['Demand [TWh/yr]'].sum()  
        
        #final_energy_europe = pd.DataFrame({'model': model, 'scenario': scenario, 'region':region_default, 'variable': 'Final Energy', 'unit': 'TWh/yr', 'year': year, 'value': [annual_demand['Demand [TWh/yr]'].sum()]})
        
        # database = pd.concat([database , final_energy_europe], ignore_index=True)
        
        # demand_sum_by_country = annual_demand.groupby('Country')['Demand [TWh/yr]'].sum()
        
        # for country in countries:
        #     final_energy_country = pd.DataFrame({'model': model, 'scenario': scenario, 'region': country, 'variable': 'Final Energy', 'unit': 'TWh/yr', 'year': year, 'value': [demand_sum_by_country.get(country, 0)]})
        #     database = pd.concat([database , final_energy_country ], ignore_index=True)
 
        
        # PRIMARY ENERGY Non Biomass RES
        
        # Calculate annual generation per node, carrier, and type
        
        # Read annual generation per node and technology
        generation_raw = network.generators_t.p.sum()*timestep*1e-6  # TWh
        generation_conventional = network.links_t.p0.sum()*timestep*1e-6 #TWh
        hydro_generation = network.storage_units_t.p.sum()*timestep*1e-6 # TWh
        
        generation = pd.concat([generation_raw, generation_conventional, hydro_generation])
        #generation = generation*twh_to_TWh # TWh/yr
        
        # Rearrange annual generation dataframe
        generation = (pd.concat([generation.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),
                    generation.index.str.extract(r'([A-Z][A-Z])', expand=True),
                    generation.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)', expand=True),
                    generation.reset_index()],ignore_index=True, axis=1).drop(3, axis=1)  # Drop third column
            .rename(columns={0: 'Node', 1: 'Country', 2: 'Source type', 4: 'Generation [TWh/yr]'}))
        
        # Put values to zero if below a certain threshold
        threshold = 1e-7
        generation = generation[generation['Generation [TWh/yr]'] > threshold]
        
        # Sum technologies from different years and clean the strings
        generation['Source type'] = generation['Source type'].str.split('-').str[0]
        generation['Source type'] = generation['Source type'].str.strip()
        generation['Source type'] = generation['Source type'].str.title()
        
        
        generation = generation.groupby(['Country','Source type'])['Generation [TWh/yr]'].sum()
        generation = generation.reset_index(drop=False)
        
        # Remove EU since it takes into account the primary resources, not electricity generation
        generation['Source type'] = generation['Source type'].replace({'Ccgt': 'CCGT'})
        generation['Source type'] = generation['Source type'].replace({'Ocgt': 'OCGT'})
        
        # Substituting nan and calling countries with a nice name
        generation['Country'] = generation['Country'].apply(get_country_name)
        generation['Country'] = generation['Country'].replace('Unknown', 'Europe') # In this case the unknow countries are aggregated for Europe

        ### ALL NON-BIOMASS RENEWABLES 
        
        filtered_generation = generation[generation['Source type'].str.contains('Onwind', case=False) | generation['Source type'].str.contains('Offwind', case=False)
                                         | generation['Source type'].str.contains('Solar', case=False) | generation['Source type'].str.contains('Ror', case=False) | generation['Source type'].str.contains('Hydro', case=False)]
        
        share_res_europe = pd.DataFrame({'model': model, 'scenario': scenario, 'region':region_default, 'variable': 'Share Non-Biomass Renewables on FE', 'unit': 'TWh/yr','year': year, 'value': [(filtered_generation['Generation [TWh/yr]'].sum()/final_energy_europe)*100]})

        database = pd.concat([database , share_res_europe], ignore_index=True)

        share_res_by_country = (filtered_generation.groupby('Country')['Generation [TWh/yr]'].sum()/annual_demand.groupby('Country')['Demand [TWh/yr]'].sum())*100
        
        
        # primary_energy_europe_res = pd.DataFrame({'model': model, 'scenario': scenario, 'region':region_default, 'variable': 'Primary Energy|Non-Biomass Renewables', 'unit': 'TWh/yr','year': year, 'value': [filtered_generation['Generation [TWh/yr]'].sum()]})
        
        # database = pd.concat([database , primary_energy_europe_res], ignore_index=True)
        
        # gen_sum_by_country_res = filtered_generation.groupby('Country')['Generation [TWh/yr]'].sum()
        
        for country in countries:
            share_res_by_country_df = pd.DataFrame({'model': model, 'scenario': scenario, 'region': country, 'variable': 'Share Non-Biomass Renewables on FE', 'unit': 'TWh/yr', 'year': year, 'value': [share_res_by_country.get(country, 0)]})
            database = pd.concat([database , share_res_by_country_df ], ignore_index=True)
            
 
    
# %% SAVING

# This will give you a duplicate error if you saved mistakenly a variable more times.
# If output in 0 everything went smoothly
# Convert nomenclature
twh_to_ej = 3.6/1000
database.loc[database['scenario'] == 'nopol', 'scenario'] = 'REF'
database.loc[database['scenario'] == 'ctax', 'scenario'] = 'CTAX'
database.loc[database['scenario'] == 'ff55', 'scenario'] = 'NZ'
database.loc[database['Unit'] == 'TWh/yr', 'value'] *= twh_to_ej



#df = df.convert_unit('TWh/yr',to='EJ/yr')

#df.data.to_csv(f'../FIDELIO_coupling/Share_RES_{folder}.csv', index=False, encoding='cp1252"')
