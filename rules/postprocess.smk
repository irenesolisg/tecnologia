# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


if config["foresight"] != "perfect":

    rule plot_power_network_clustered:
        params:
            plotting=config_provider("plotting"),
        input:
            network=resources("networks/base_s_{clusters}.nc"),
            regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            map=resources("maps/power-network-s-{clusters}.pdf"),
        threads: 1
        resources:
            mem_mb=4000,
        benchmark:
            benchmarks("plot_power_network_clustered/base_s_{clusters}")
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network_clustered.py"

    rule plot_power_network:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_power_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_power_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network.py"

    rule plot_hydrogen_network:
        params:
            plotting=config_provider("plotting"),
            foresight=config_provider("foresight"),
        input:
            network=RESULTS
            + "postnetworks/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-h2_network_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_hydrogen_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_hydrogen_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_hydrogen_network.py"

    rule plot_gas_network:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            map=RESULTS
            + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-ch4_network_{planning_horizons}.pdf",
        threads: 2
        resources:
            mem_mb=10000,
        log:
            RESULTS
            + "logs/plot_gas_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.log",
        benchmark:
            (
                RESULTS
                + "benchmarks/plot_gas_network/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}"
            )
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_gas_network.py"


if config["foresight"] == "perfect":

    def output_map_year(w):
        return {
            f"map_{year}": RESULTS
            + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_"
            + f"{year}.pdf"
            for year in config_provider("scenario", "planning_horizons")(w)
        }

    rule plot_power_network_perfect:
        params:
            plotting=config_provider("plotting"),
        input:
            network=RESULTS
            + "postnetworks/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_brownfield_all_years.nc",
            regions=resources("regions_onshore_base_s_{clusters}.geojson"),
        output:
            unpack(output_map_year),
        threads: 2
        resources:
            mem_mb=10000,
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/plot_power_network_perfect.py"


rule make_summary:
    params:
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        scenario=config_provider("scenario"),
        RDIR=RDIR,
    input:
        networks=expand(
            RESULTS
            + "postnetworks/base_s_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            allow_missing=True,
        ),
        costs=lambda w: (
            resources("costs_{}.csv".format(config_provider("costs", "year")(w)))
            if config_provider("foresight")(w) == "overnight"
            else resources(
                "costs_{}.csv".format(
                    config_provider("scenario", "planning_horizons", 0)(w)
                )
            )
        ),
        ac_plot=expand(
            resources("maps/power-network-s-{clusters}.pdf"),
            **config["scenario"],
            allow_missing=True,
        ),
        costs_plot=expand(
            RESULTS
            + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-costs-all_{planning_horizons}.pdf",
            **config["scenario"],
            allow_missing=True,
        ),
        h2_plot=lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-h2_network_{planning_horizons}.pdf"
                if config_provider("sector", "H2_network")(w)
                else []
            ),
            **config["scenario"],
            allow_missing=True,
        ),
        ch4_plot=lambda w: expand(
            (
                RESULTS
                + "maps/base_s_{clusters}_l{ll}_{opts}_{sector_opts}-ch4_network_{planning_horizons}.pdf"
                if config_provider("sector", "gas_network")(w)
                else []
            ),
            **config["scenario"],
            allow_missing=True,
        ),
    output:
        nodal_costs=RESULTS + "csvs/nodal_costs.csv",
        nodal_capacities=RESULTS + "csvs/nodal_capacities.csv",
        nodal_cfs=RESULTS + "csvs/nodal_cfs.csv",
        cfs=RESULTS + "csvs/cfs.csv",
        costs=RESULTS + "csvs/costs.csv",
        capacities=RESULTS + "csvs/capacities.csv",
        curtailment=RESULTS + "csvs/curtailment.csv",
        energy=RESULTS + "csvs/energy.csv",
        supply=RESULTS + "csvs/supply.csv",
        supply_energy=RESULTS + "csvs/supply_energy.csv",
        nodal_supply_energy=RESULTS + "csvs/nodal_supply_energy.csv",
        prices=RESULTS + "csvs/prices.csv",
        weighted_prices=RESULTS + "csvs/weighted_prices.csv",
        market_values=RESULTS + "csvs/market_values.csv",
        price_statistics=RESULTS + "csvs/price_statistics.csv",
        metrics=RESULTS + "csvs/metrics.csv",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/make_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/make_summary.py"


rule plot_summary:
    params:
        countries=config_provider("countries"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        emissions_scope=config_provider("energy", "emissions"),
        plotting=config_provider("plotting"),
        foresight=config_provider("foresight"),
        co2_budget=config_provider("co2_budget"),
        sector=config_provider("sector"),
        RDIR=RDIR,
    input:
        costs=RESULTS + "csvs/costs.csv",
        energy=RESULTS + "csvs/energy.csv",
        balances=RESULTS + "csvs/supply_energy.csv",
        eurostat="data/eurostat/Balances-April2023",
        co2="data/bundle/eea/UNFCCC_v23.csv",
    output:
        costs=RESULTS + "graphs/costs.svg",
        energy=RESULTS + "graphs/energy.svg",
        balances=RESULTS + "graphs/balances-energy.svg",
    threads: 2
    resources:
        mem_mb=10000,
    log:
        RESULTS + "logs/plot_summary.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_summary.py"


STATISTICS_BARPLOTS = [
    "capacity_factor",
    "installed_capacity",
    "optimal_capacity",
    "capital_expenditure",
    "operational_expenditure",
    "curtailment",
    "supply",
    "withdrawal",
    "market_value",
]

STATISTICS = {
    "capacity_factor": ("-", "p.u."),
    "installed_capacity": (1e3, "GW"),
    "optimal_capacity": (1e3, "GW"),
    "capex": (1e9, "bn €"),
    "opex": (1e9, "bn €"),
    "total_cost": ("1e9", "bn €"),
    "curtailment": (1e3, "GWh"),
    "supply": (1e6, "TWh"),
    "withdrawal": (1e6, "TWh"),
    "energy_balance": (1e6, "TWh"),
    "market_value": ("-", "€/MWh"),
}


rule write_statistics:
    params:
        statistics=STATISTICS,
    input:
        network=RESULTS
        + "postnetworks/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        **{
            f"{metric}": RESULTS
            + "statistics/csv/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/{carrier}_"
            + f"{metric}.csv"
            for carrier in config_provider("plotting", "statistics")(run).get(
                "carriers", "all"
            )
            for metric in config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            )
        },
        csv_touch=RESULTS
        + "statistics/csv/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/.statistics_{carrier}_csv",
    log:
        RESULTS
        + "logs/write_statistics/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_country-{country}_carrier-{carrier}.log",
    script:
        "../scripts/write_statistics.py"


rule plot_statistics_single:
    params:
        plotting=config_provider("plotting"),
        statistics=STATISTICS,
    input:
        **{
            f"{metric}": RESULTS
            + "statistics/csv/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/{carrier}_"
            + f"{metric}.csv"
            for carrier in config_provider("plotting", "statistics")(run).get(
                "carriers", "all"
            )
            for metric in config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            )
        },
    output:
        **{
            f"{metric}": RESULTS
            + "statistics/figures/single/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/{carrier}_"
            + f"{metric}.pdf"
            for carrier in config_provider("plotting", "statistics")(run).get(
                "carriers", "all"
            )
            for metric in config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            )
        },
        barplots_touch=RESULTS
        + "statistics/figures/single/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/.statistics_{carrier}_plots",
    log:
        RESULTS
        + "logs/plot_statistics_single/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}_country-{country}_carrier-{carrier}.log",
    script:
        "../scripts/plot_statistics_single.py"


rule plot_statistics_comparison:
    params:
        plotting=config_provider("plotting"),
        statistics=STATISTICS,
    input:
        expand(
            RESULTS
            + "statistics/csv/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/{carrier}_{metric}.csv",
            **config["scenario"],
            metric=config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            ),
            allow_missing=True,
        ),
    output:
        **{
            f"{metric}": RESULTS
            + "statistics/figures/comparison/country_{country}/{carrier}_"
            + f"{metric}.pdf"
            for carrier in config_provider("plotting", "statistics")(run).get(
                "carriers", "all"
            )
            for metric in config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            )
        },
        barplots_touch=RESULTS
        + "statistics/figures/comparison/country_{country}/.statistics_{carrier}_plots",
    log:
        RESULTS
        + "logs/plot_statistics_comparison/country-{country}_carrier-{carrier}.log",
    script:
        "../scripts/plot_statistics_comparison.py"


def get_scnario_copmarison_run(w):
    run = config_provider("plotting", "statistics")(w).get(
        "scenario_comparison", config["run"]["name"]
    )
    if run == [""] or run == "":
        run = config["run"]["name"]
    return run


rule plot_statistics_scenario_comparison:
    params:
        plotting=config_provider("plotting"),
        statistics=STATISTICS,
    input:
        expand(
            RESULTS
            + "statistics/csv/elec_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}/country_{country}/{carrier}_{metric}.csv",
            **config["scenario"],
            metric=config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            ),
            run=get_scnario_copmarison_run(run),
            allow_missing=True,
        ),
    output:
        **{
            f"{metric}": "results/statistics/"
            + config_provider("plotting", "statistics")(run).get(
                "comparison_folder", ""
            )
            + "/"
            + "figures/country_{country}/{carrier}_"
            + f"{metric}.pdf"
            for carrier in config_provider("plotting", "statistics")(run).get(
                "carriers", "all"
            )
            for metric in config_provider("plotting", "statistics")(run).get(
                "metrics", STATISTICS
            )
        },
        barplots_touch="results/statistics/"
        + config_provider("plotting", "statistics")(run).get(
            "comparison_folder", "results/scenario_comparison"
        )
        + "/"
        + "figures/country_{country}/.statistics_{carrier}_plots",
    log:
        "results/statistics/"
        + config_provider("plotting", "statistics")(run).get(
            "comparison_folder", "scenario_comparison"
        )
        + "/"
        + "logs/plot_statistics_scenario_comparison/country-{country}_carrier-{carrier}.log",
    script:
        "../scripts/plot_statistics_comparison.py"


rule plot_base_statistics:
    params:
        plotting=config_provider("plotting"),
        barplots=STATISTICS_BARPLOTS,
    input:
        network=RESULTS + "networks/base_s_{clusters}_elec_l{ll}_{opts}.nc",
    output:
        **{
            f"{plot}_bar": RESULTS
            + f"figures/statistics_{plot}_bar_base_s_{{clusters}}_elec_l{{ll}}_{{opts}}.pdf"
            for plot in STATISTICS_BARPLOTS
        },
        barplots_touch=RESULTS
        + "figures/.statistics_plots_base_s_{clusters}_elec_l{ll}_{opts}",
    script:
        "../scripts/plot_statistics.py"
