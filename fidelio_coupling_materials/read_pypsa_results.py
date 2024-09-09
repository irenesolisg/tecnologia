# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:14:43 2023

@author: alice
"""

#%% Import the libraries needed

import pypsa

import os

import netCDF4

# from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
import plotly.offline as pltly

from matplotlib import pyplot as plt
from matplotlib import pylab


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

pltly.init_notebook_mode(connected=True)

#%% File nc
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
os.path.abspath("elec_s_37_lv1.5_Co2L0p45-3H-T-H-B-I-A-solar+p3-dist1_2030.nc")

filename = r"C:\Users\alice\Desktop\CMCC\PyPSA material\results\July23\elec_s_37_lv1.5__Co2Lff55-219H-T-H-B-I-A-solar+p3-dist1_2030.nc"
filename2 = r"C:\Users\alice\Desktop\CMCC\PyPSA material\results\ff55_1H\postnetworks\elec_s_37_lv1.5__Co2L0p45-1H-T-H-B-I-A-solar+p3-dist1_2030.nc"
# with netCDF4.Dataset(filename) as dataset:
#     print(dataset.variables)

network = pypsa.Network(filename)
network2 = pypsa.Network(filename2)

# Read the variables in the newtwork
variables = dir(network)

#%% Save some data

aconstraints = network.global_constraints
abuses = network.buses
agenerators = network.generators
abranches = network.branches
acarriers = network.carriers
alinks = network.links
aloads = network.loads
astores = network.stores
astorage = network.storage_units
filtered_co2 = astores[astores.index.str.contains('co2', case=False)]

#%% Get energy prices
mprice = network.buses_t.marginal_price
elec_prices = mprice[[col for col in mprice.columns if "low voltage" in col]]
elec_prices = elec_prices.rename(columns=lambda x: x.replace('low voltage', '').strip())


mprice_default = network2.buses_t.marginal_price
elec_prices_default = mprice_default[[col for col in mprice_default.columns if "low voltage" in col]]
elec_prices_default = elec_prices_default.rename(columns=lambda x: x.replace('low voltage', '').strip())

# Get electricity loads per hours

loads = network.loads_t.p
elec_loads = loads[[col for col in loads.columns if col.endswith('0')]]

loads_default = network2.loads_t.p
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

changes = ((weighted_average-weighted_average_default)/weighted_average_default)*100

changes.to_csv('perc_changes_elec_prices.csv', index=True)




network.plot();

for c in network.iterate_components(list(network.components.keys())[2:]):
    print("Component '{}' has {} entries".format(c.name, len(c.df)))
    
network.snapshots[:10]
len(network.snapshots)
lines = network.lines
generators = network.generators
network.storage_units.head()
network.loads_t.p_set.head()
network.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))
network.generators_t.p_max_pu.head()
network.generators_t.p_max_pu.loc["2013-01","IT1 0 offwind-ac"].plot(figsize=(15,3));

network.objective/1e9 #billion euros per year


gen_daily=network.generators_t.p*24  #MWh

gen_yearly=network.generators_t.p.sum()*24/1000000     #TWh

gen_daily.to_excel('Gen_daily.xlsx')
gen_yearly.to_excel('Gen_yearly_nod.xlsx')

ex_hist = pd.ExcelFile("Gen_yearly_nod.xlsx")
df = ex_hist.parse("Sheet1")
print(df)

# df.assign(Countries=df['name'].str[:3])
df['Countries']= df['Generator'].astype(str).str[0:2]
df['nodes']= df['Generator'].astype(str).str[0:3]
df['Country  num']= df['Generator'].astype(str).str[2:3]
df['tech']= df['Generator'].astype(str).str[6:]
print(df)

    
nodes=df['nodes'].unique()
countries=df['Countries'].unique()
tech=df['tech'].unique()


# dt=pd.DataFrame()
dk=pd.DataFrame(index=tech)

for a in range(len(nodes)):
    print(df[df['nodes']==nodes[a]])
    dp=df[df['nodes']==nodes[a]]
    dp=dp.set_index('tech')
    
    dk[nodes[a]]=dp[0]
    
print(dk.T)

dk=dk.fillna(0)    
dk=dk.T

i=0
case=[]
while i < len(nodes)-1:
    if dk.index.astype(str).str[0:2][i]==dk.index.astype(str).str[0:2][i+1]:
        dk.iloc[i,:]=dk.iloc[i,:]+dk.iloc[i+1,:]
        case.append(i+1)
        i+=1
    i+=1

for b in case:
    dk=dk.drop(nodes[b])

dk.index=dk.index.astype(str).str[0:2]

dk=dk.drop('EU')



dk.to_excel('Gen_yearly_nat.xlsx')

dic_col={'offwind-ac': '#C3C29D',
         'onwind': '#EB8627', 
         'solar': '#F9DB61',
         'solar rooftop': '#ECBF48', 
         'residential rural solar thermal collector': '#EB8627',
         'services rural solar thermal collector': '#231F20', 
         'ror': '#9CC5E7',
         'residential urban decentral solar thermal collector': '#A0B485', 
         'services urban decentral solar thermal collector': '#E6E7E8',
         'offwind-dc': '#663E90',
         'urban central solar thermal collector': '#9A91C4',
         'gas': '#989898',
         '': '#FFFFFF',
         'geothermal': '#9C3121',
         }

x=dk.index
y=[]
for g in range(33):
    y.append(1000*g)



colors_list=[]
for a in range(len(dk.columns)):
    colors_list.append(dic_col[dk.columns[a]])
    
# col_infr=[ '#231F20', '#BBBDBF', '#E6E7E8', '#C3C29D', '#497441', '#9C3121','#F9DB61', '#ECBF48','#EB8627','#9CC5E7' , '#3A7FAD']
#col_fue= ['powderblue', 'saddlebrown', 'peru', 'darkred', 'khaki', 'tan', 'lime']
fig, ax = plt.subplots(figsize=(12,7))

plt.rcParams["font.family"] = "Calibri"
dk.plot.bar(stacked=True, color=colors_list, ax=ax, alpha=1, edgecolor='none', rot=0, width=0.6)#, position=1

# axes2 = plt.twinx()
# axes2=plot(x,y)
# dk['sum'].plot(ax=ax, color="#E15837", label="Electricity demand 2050", linestyle='--', linewidth=2.5)

handles, labels = ax.get_legend_handles_labels()

legend1 = ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1, title="Legend", fontsize=12)
#legend2 = ax.legend(reversed(handles[len(infrastructure):]), reversed(fuels), bbox_to_anchor=(1.01, 0.20), loc='center left', ncol=1, title="Legend fuels")
ax.add_artist(legend1)
#ax.add_artist(legend2)
# ax.set_ylim(0, 12)
ax.set_ylabel('Electricity generation and demand [TWh]', fontsize=14)
ax.set_xlabel('', fontsize=10)
# ax.set_xticklabels(['Baseline\n2015', 'PNIEC\n2030', 'Advanced\n2030'], fontsize=12)

plt.grid(linestyle='dotted')
plt.show()

pylab.savefig("Electricity_mix_national.png", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)
#pylab.savefig("Electricity_mix.svg", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)




# load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum().reindex(network.buses.index,fill_value=0.)

#matplotlib: https://pypsa.org/examples/scigrid-lopf-then-pf.html

# fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree()})

# fig.set_size_inches(10,10)

# load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

# network.plot(bus_sizes=0.00005*load_distribution,ax=ax, title="Load distribution")
# plt.show()
# pylab.savefig("map.png", bbox_inches="tight", dpi=300)



# Plotly: https://pypsa.org/examples/scigrid-lopf-then-pf-plotly.html
# fig = dict(data=[],layout=dict(width=700,height=700))

# fig = network.iplot(bus_sizes=0.001*load_distribution, fig=fig,
#                       bus_text='Load at bus ' + network.buses.index + ': ' + round(load_distribution).values.astype(str) + ' MW',
#                       title="Load distribution",
#                       line_text='Line ' + network.lines.index)
# plot(fig)
# fig.write_image("fig11.png")



#%% Generation and Demand

gen_total = network.generators_t.p.sum()*24/1000000 #TWh
gen_daily = network.generators_t.p*24 #MWh
gen_total.to_excel('Generation_annual.xlsx')
gen_daily.to_excel('Generation_daily.xlsx')

ex_hist = pd.ExcelFile("Generation_annual.xlsx")
df = ex_hist.parse("Sheet1")
print(df)

df['countries'] = df['Generator'].astype(str).str[0:2]
df['nodes'] = df['Generator'].astype(str).str[0:3]
df['country  number'] = df['Generator'].astype(str).str[2:3]
df['tech'] = df['Generator'].astype(str).str[6:]
print(df)

nodes = df['nodes'].unique()
tech = df['tech'].unique()
countries = df['countries'].unique()

# Organize by countries and tech

dk=pd.DataFrame(index=tech)

for a in range(len(nodes)):
    dp=df[df['nodes']==nodes[a]]
    dp=dp.set_index('tech') 
    dk[nodes[a]]=dp[0]
    dk=dk.fillna(0)    
    
print(dk.T)

dk=dk.T
i=0
case=[]

while i < len(nodes)-1:
    if dk.index.astype(str).str[0:2][i] == dk.index.astype(str).str[0:2][i+1]: 
        dk.iloc[i,:] = dk.iloc[i,:] + dk.iloc[i+1,:]
        case.append(i+1)
        i+=1
       
    i+=1         
       
for b in case:
    dk=dk.drop(nodes[b])
dk.index = dk.index.astype(str).str[0:2]
print(dk.T.sum())  

dk.to_excel('Generation_annual_per_country.xlsx')

#%% Demand

demand_total = network.loads_t.p.sum()*24/1000000 #TWh
demand_daily = network.loads_t.p*24 #MWh
# PyPSA averages the required daily power so we *24 for the total_daily_demand
demand_total.to_excel('Demand_annual.xlsx')
demand_daily.to_excel('Demand_daily.xlsx')

ex_hist = pd.ExcelFile("Demand_annual.xlsx")
df_demand = ex_hist.parse("Sheet1")
print(df_demand)

df_demand['nodes'] = df_demand['Load'].astype(str).str[0:3]
nodes = df_demand['nodes'].unique()
df_demand['demand'] = df_demand[0].astype(float)
print(df_demand)

dm = pd.DataFrame(index = nodes, columns=('demand','generation'))

for i in range(len(nodes)):
    dm.demand[i] = df_demand.demand[i]
    


i=0
case=[]
while i < len(nodes)-1:
    if dm.index.astype(str).str[0:2][i] == dm.index.astype(str).str[0:2][i+1]: 
        dm.iloc[i] = dm.iloc[i] + dm.iloc[i+1]
        case.append(i+1)
        i+=1
    i+=1         
       
for b in case:
    dm=dm.drop(nodes[b])

for i in range(len(nodes)-len(case)):
    dm.generation[i]=dk.T.sum()[i]


dm.index = dm.index.astype(str).str[0:2]
print(dm)
print(dm.sum())  

dm.to_excel('Demand_and_generation_per_country.xlsx')


# Plots

dic_col={'offwind-ac': '#C44A21', 'onwind': '#EB8627', 
          'solar': '#F9DB61', 'CCGT': '#BBBDBF', 
          'OCGT': '#9C9EA1', 'coal': '#231F20', 
          'ror': '#9CC5E7', 'biomass': '#497441', 
          'nuclear': '#EA85C7', 'offwind-dc': '#C43221',
          'lignite': '#8C8C8C', 'oil': '#404041', 
          'geothermal': '#9C3121'}


colors_list=[]
for a in range(len(dk.columns)):
    colors_list.append(dic_col[dk.columns[a]])
    
fig, ax = plt.subplots(figsize=(12,7))

color_demand = []
dic_col_demand = {'demand': '#E15837'}
color_demand.append(dic_col_demand[dm.columns[0]])

dk.plot.bar(stacked=True, color=colors_list, ax=ax, alpha=1, edgecolor='none', rot=0, width=0.6)#, position=1
dm['demand'].plot(ax=ax, color=color_demand, label="Electricity demand 2050", linestyle='--', linewidth=2.5)

handles, labels = ax.get_legend_handles_labels()

legend1 = ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1, title="Legend", fontsize=12)
ax.add_artist(legend1)

ax.set_ylabel('Electricity generation and demand [TWh]', fontsize=14)
ax.set_xlabel('European countries', fontsize=10)

plt.grid(linestyle='dotted')
plt.show()

pylab.savefig("Electricity_mix_national.png", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)
#pylab.savefig("Electricity_mix.svg", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)

#%% Input: CO2 - emissions constraints and annual emissions


CO2_global_limit = network.global_constraints['constant'] #CO2_eq tonn / year
CO2_shadow_price = network.global_constraints['mu'] # €/MWh
CO2_factors = network.carriers.co2_emissions #CO2_eq tonn / MWh


gen_total = network.generators_t.p.sum()*24 #MWh
gen_total.to_excel('Generation_annual_MWh.xlsx')

ex_hist = pd.ExcelFile("Generation_annual_MWh.xlsx")
de = ex_hist.parse("Sheet1")
print(de)

de['emissions'] = 0

de['generation'] = de[0]
de['countries'] = de['Generator'].astype(str).str[0:2]
de['nodes'] = de['Generator'].astype(str).str[0:3]
de['country  number'] = de['Generator'].astype(str).str[2:3]
de['tech'] = de['Generator'].astype(str).str[6:]
print(de)

emiss = []

for i in range(len(de.index)):
    for j in range(len(CO2_factors)):
        if de.tech[i] == CO2_factors.index[j]:
            emiss.append(de.generation[i] * CO2_factors[j])
            
de['emissions']=emiss
print(de)

nodes = de['nodes'].unique()
tech = de['tech'].unique()
countries = de['countries'].unique()

dk = pd.DataFrame(index = tech)

for a in range(len(nodes)):
    dem=de[de['nodes']==nodes[a]]
    dem=dem.set_index('tech') 
    dk[nodes[a]]=dem['emissions']
    dk=dk.fillna(0)  

print(dk.T)
dk=dk.T
i=0
case=[]
while i < len(nodes)-1:
    if dk.index.astype(str).str[0:2][i] == dk.index.astype(str).str[0:2][i+1]: 
        dk.iloc[i,:] = dk.iloc[i,:] + dk.iloc[i+1,:]
        case.append(i+1)
        i+=1
       
    i+=1         
       
for b in case:
    dk=dk.drop(nodes[b])
dk.index = dk.index.astype(str).str[0:2]
print(dk)  


dk.to_excel('Emissions_per_country.xlsx')

dic_col={'offwind-ac': '#C44A21', 'onwind': '#EB8627', 
          'solar': '#F9DB61', 'CCGT': '#BBBDBF', 
          'OCGT': '#9C9EA1', 'coal': '#231F20', 
          'ror': '#9CC5E7', 'biomass': '#497441', 
          'nuclear': '#EA85C7', 'offwind-dc': '#C43221',
          'lignite': '#8C8C8C', 'oil': '#404041', 
          'geothermal': '#9C3121'}


colors_list=[]
for a in range(len(dk.columns)):
    colors_list.append(dic_col[dk.columns[a]])
    
fig, ax = plt.subplots(figsize=(12,7))

dk.plot.bar(stacked=True, color=colors_list, ax=ax, alpha=1, edgecolor='none', rot=0, width=0.6)#, position=1

handles, labels = ax.get_legend_handles_labels()

legend1 = ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1, title="Legend", fontsize=12)
ax.add_artist(legend1)
ax.set_ylabel('CO2 emissions per country [CO2_eq tonn / year]', fontsize=14)
ax.set_xlabel('European countries', fontsize=10)

plt.grid(linestyle='dotted')
plt.show()

pylab.savefig("CO2_emissions_national.png", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)
#pylab.savefig("Electricity_mix.svg", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)

#grafico a torta per percentuali ogni paese
fig, bx = plt.subplots(figsize=(12,7))
casoit = dk.T.IT / dk.T.IT.sum() * 100

x = np.array(casoit)

case1 = []
case2 = []
for i in range(len(casoit)):
    if casoit[i]==0:
        case1.append(i)
        
    else:
        case2.append(casoit.index[i])
x = np.delete(x, case1)

labels = case2
plt.pie(x, explode=None, labels = labels, colors = None, autopct = None)


total_perc = dk.T.sum() / dk.T.sum().sum() * 100
plt.pie(total_perc, explode=None, labels = dk.index, colors = None, autopct = None)




#%% RES potential and profiles

p_nom_opt = network.generators.p_nom_opt

p_max = network.generators.p_nom_max

p_min = network.generators.p_nom

p_sum = network.generators_t.p.sum()

# these are the profiles of maximum power per unit
profiles = network.generators_t.p_max_pu

# here I compute the profiles starting from generators_t.p / p_nom_opt

a=network.generators.p_nom_opt.T
a.to_excel("P_nom_per_node_per_tech.xlsx")
b = network.generators_t.p
c=pd.DataFrame(index=b.index, columns=b.columns)
names=b.columns

for i in range(len(b.columns)):
    x = np.array(b[names[i]])
    c[names[i]] = x / a[i]

ren_c = pd.DataFrame(index = c.index, columns=profiles.columns)

for i in range(len(c.columns)):
    for j in range(len(profiles.columns)):
        if c.columns[i] == profiles.columns[j]:
            ren_c[c.columns[i]]=c[c.columns[i]]

# now the 2 arrays (profiles and ren_c) show the profiles of renewables
# they are not quite the same ..  


# equivalent hours in the two ways to make comparison
h_eq_profiles = profiles.sum()*24
h_eq_ren_c = ren_c.sum()*24
# to check why slightly different


## check network.generators_t diviso potenza nominale 
## salvare p_nom per ogni nodo le tech -> excel salvarle!! 
## profili normalizzati 0 e 1 + ore equivalenti
# confronto con p_nom_max file csv
## i profili vanno a zero quando no prod e a 1 quando potenza nom raggiunta
## strano il profilo 0.1-0.2

# strano però cmq è mediato su 24 h ci sta sia cosi basso

## somma profilo 0 e 1 cosi H_eq 1200 italia e sono di piu di svezia e norvegia check!
# salvare dataframe in excel con questi valori profiles.sum()*24
# come l'input influenza l'output
# potenza installata inizio. installata fine. e poi l'ottimizzazione in mezzo alla barra
# se troviamo i numeri potrebbe essere utile sito GSE come confronto

# potenza installata all'inizio per rinnovabili è zero (p_nom che è quello al 2013) ma non è vero!

it1 = pd.DataFrame(index = ren_c.index, columns = ('IT0','IT1'))

it1.IT1 = c['IT1 0 solar']
it1.IT3 = c['IT3 0 solar']

profiles.to_excel('VRES_profiles.xlsx')

it = pd.DataFrame(index = profiles.index, columns = ('IT1','IT3')) #just solar production

it.IT0 = profiles['IT1 0 solar']

it.IT1 = profiles['IT3 0 solar']

fig, ax = plt.subplots(figsize=(12,7))

it.plot(ax=ax, color=['blue','red'], label="Solar profiles potential 2013", linestyle='-', linewidth=1.5)

handles, labels = ax.get_legend_handles_labels()

legend1 = ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1, title="Legend", fontsize=12)
ax.add_artist(legend1)

ax.set_ylabel('Potential [-]', fontsize=14)
ax.set_xlabel('', fontsize=10)

plt.grid(linestyle='dotted')
plt.show()

pylab.savefig("Italian_solar_variable_profile.png", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)
#pylab.savefig("Electricity_mix.svg", bbox_extra_artists=[legend1], bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(figsize=(12,7))
it.IT0.plot(ax=ax, color=['blue'], label="Solar profiles potential 2013", linestyle='-', linewidth=1.5)
# it1.IT0.plot(ax=ax, color=['red'], label="Solar profiles potential 2013", linestyle='-', linewidth=1.5)


# check su dk wind

dk = pd.DataFrame(index = ren_c.index, columns = ('DK0_ren_c','DK0_profiles'))
dk.DK0_ren_c=ren_c['DK0 0 onwind']
dk.DK0_profiles=profiles['DK0 0 onwind']

fig, ax = plt.subplots(figsize=(12,7))

dk.plot(ax=ax, color=['blue','red'], label="Solar profiles potential 2013", linestyle='-', linewidth=1.5)

# check su AL0 offwind-ac

al = pd.DataFrame(index = ren_c.index, columns = ('AL0_ren_c','AL0_profiles'))
al.AL0_ren_c=ren_c['AL0 0 offwind-ac']
al.AL0_profiles=profiles['AL0 0 offwind-ac']

fig, ax = plt.subplots(figsize=(12,7))

al.plot(ax=ax, color=['blue','red'], label="Solar profiles potential 2013", linestyle='-', linewidth=1.5)





#%% Storage and storage units
stores = network.stores #H2 and batteries
storage_units = network.storage_units #PHS and hydro

max_hours = storage_units.max_hours
pnom = storage_units.p_nom_opt #same as p_nom in input because it is not extendable
marginal_cost_hydros = storage_units.marginal_cost
capital_cost_hydros = storage_units.capital_cost

lifetime=80

generation = network.storage_units_t.p # p>0 if net generation

gen_from_hydro=generation.sum().sum()*24/(10**6) #TWh
gen_from_batH2 = network.stores_t.p.sum().sum()*24/(10**6)

gen_supertotal = dm['generation'].sum()+gen_from_batH2+gen_from_hydro
print(gen_supertotal)
print(dm['demand'].sum())

#%%
#Cost comparison for each bus: what costs less?
timestep = 1
marginal_cost_gen = network.generators.marginal_cost #€/MWh
capital_cost_gen  = network.generators.capital_cost #€/MW
h_eq_gen = (network.generators_t.p.sum()*timestep)/(network.generators.p_nom_opt)

table = pd.DataFrame(index=marginal_cost_gen.index, columns=('CAPEX [€/MW]','OPEX [€/MWh]','Eq hours [h/y]','Lifetime [y]','LCOE [€/MWh]')) #retrieved from outputs


table['CAPEX [€/MW]']=capital_cost_gen

table['OPEX [€/MWh]']=marginal_cost_gen
table['Eq hours [h/y]']=h_eq_gen

for i in range(len(table.index)):
    for j in range(len(dl.index)):
        if table.index.astype(str).str[5][i]==' ': #caso nodi<10
            if table.index.astype(str).str[6:][i]==dl.index[j]:
                table['Lifetime [y]'][i]=dl.lifetime[j]
        elif table.index.astype(str).str[6][i]==' ': #caso almeno 10 nodi
            if table.index.astype(str).str[7:][i]==dl.index[j]:
                table['Lifetime [y]'][i]=dl.lifetime[j]



LCOE=(table['CAPEX [€/MW]']/table['Lifetime [y]']+table['OPEX [€/MWh]'])/table['Eq hours [h/y]']
            
table['LCOE [€/MWh]']=LCOE   

#we put the marginal price for fossil fuels which have capex=0 since they are not extendable and their cost is considered sinked
for i in range(len(table.index)):
    if table['CAPEX [€/MW]'][i]==0:
        table['LCOE [€/MWh]'][i]=table['OPEX [€/MWh]'][i]





#%%
# potentials

potential = network.generators.p_nom_max

potential.to_excel('Potential_all_tech.xlsx')


#%% Costs

gen_total = network.generators_t.p.sum()*24 #MWh
gen_total.to_excel('Generation_annual_MWh.xlsx')

ex_hist = pd.ExcelFile("Generation_annual_MWh.xlsx")
de = ex_hist.parse("Sheet1")

de['tech'] = de['Generator'].astype(str).str[6:] 
tech = de['tech'].unique()

marginal_cost = network.generators.marginal_cost #$/MWh
capital_cost  = network.generators.capital_cost  #$/MW

clearing_price = network.buses_t.marginal_price #$/MWh


life = pd.read_csv("costs.csv")
life.index=life.technology
lifetime = pd.DataFrame(index = life.index, columns = ('technology','lifetime'))

for i in range(len(life)): 
    if life.parameter[i] == 'lifetime':
        # if life.technology[i] == tech:
            lifetime.technology[i] = life.technology[i]
            lifetime.lifetime[i] = life.value[i]
            
dl = pd.DataFrame(index=tech, columns = ('lifetime','0') )
   
for i in range(len(lifetime)):
    for j in range(len(tech)):
        if lifetime.technology[i] == tech[j]:
            dl.lifetime[j]=lifetime.lifetime[i]

del dl['0'] 


#%% Links

# link_capacity = network.links

# p_out_node1_links = network.links_t.p1

bus0=[]

for i in range(len(network.links.carrier)):
    if network.links.carrier[i] == 'DC':
        bus0.append(network.links.bus0[i])

bus1=[]

for i in range(len(network.links.carrier)):
    if network.links.carrier[i] == 'DC':
        bus1.append(network.links.bus1[i])
        
    
col=[]

for i in range(len(bus0)):
    col.append(bus0[i] + ' -> ' + bus1[i])
  
    
p_out_node0_links_ugly = network.links_t.p0
p_out_node0_links = pd.DataFrame(index=p_out_node0_links_ugly.index, columns=col)

nominal_capacity_links = []


for i in range(len(p_out_node0_links_ugly)):
    if i < len(bus0):
        p_out_node0_links[p_out_node0_links.columns[i]] = p_out_node0_links_ugly[p_out_node0_links_ugly.columns[i]]
        nominal_capacity_links.append(network.links.p_nom_opt[i])


scarto = 0.01 # 1 per cento

bottlenecks = pd.DataFrame(index=p_out_node0_links.index, columns=col)

for i in range(len(nominal_capacity_links)):
    for j in range(len(p_out_node0_links)):
        if p_out_node0_links.iloc[j][i] >= nominal_capacity_links[i] * (1 - scarto):
            bottlenecks.iloc[j][i] = 'Saturated'
        else:
            bottlenecks.iloc[j][i] = 'Not saturated'

# Lines


# lines_capacity = network.lines
   
col_lines=[]

for i in range(len(network.lines)):
    col_lines.append(network.lines.bus0[i] + ' -> ' + network.lines.bus1[i])



p_out_node0_lines = pd.DataFrame(index=network.lines_t.p0.index, columns=col_lines)
nominal_capacity_lines = []

for i in range(len(network.lines_t.p0.columns)):
        p_out_node0_lines[p_out_node0_lines.columns[i]] = network.lines_t.p0
        nominal_capacity_lines.append(network.lines.s_nom_opt[i])


scarto = 0.01 # 1 per cento

bottlenecks_lines = pd.DataFrame(index=p_out_node0_lines.index, columns=col_lines)

for i in range(len(nominal_capacity_lines)):
    for j in range(len(p_out_node0_lines)):
        if p_out_node0_lines.iloc[j][i] >= nominal_capacity_lines[i] * (1 - scarto):
            bottlenecks_lines.iloc[j][i] = 'Saturated'
        else:
            bottlenecks_lines.iloc[j][i] = 'Not saturated'




# load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum().reindex(network.buses.index,fill_value=0.)

# #matplotlib: https://pypsa.org/examples/scigrid-lopf-then-pf.html

# fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree()})

# fig.set_size_inches(10,10)

# load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()

# network.plot(bus_sizes=0.00005*load_distribution,ax=ax, title="Load distribution")
# plt.show()
# pylab.savefig("map.png", bbox_inches="tight", dpi=300)



# Plotly: https://pypsa.org/examples/scigrid-lopf-then-pf-plotly.html
# fig = dict(data=[],layout=dict(width=700,height=700))

# fig = network.iplot(bus_sizes=0.001*load_distribution, fig=fig,
#                       bus_text='Load at bus ' + network.buses.index + ': ' + round(load_distribution).values.astype(str) + ' MW',
#                       title="Load distribution",
#                       line_text='Line ' + network.lines.index)
# plot(fig)
# fig.write_image("fig11.png")

#%% Horizontal bar chart



