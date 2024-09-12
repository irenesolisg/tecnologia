#! /bin/sh
# queues: g_ (GPU usage), s_ (serial jobs), p_ (parallel jobs);  short, medium, long refer to the maximum lenght after which the job is stopped

OPTS='gdxcompress=1 errmsg=1 --verbose=1'

PRJ='CBAM'
VERS='2'
BRANCH='master'

BSUB1='bsub -n 1 -q p_short -M 32G -P 0607 -Is'
BSUB2='bsub -n 1 -q s_short -M 32G -P 0607 -I'
PYPSA='snakemake -call all --configfile=config/fidelio_config.yaml'
FIDEL='gams run_fidelio.gms --dataType=c --set_split=figaroD35'


display_usage() {
        echo "This script launches the PyPSA-Eur and FIDELIO coupling runs."
        echo -e "\nUsage:\n run_pypsa_fide_link.sh"
}


${BSUB1} ${PYPSA}
