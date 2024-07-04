# A script to setup Vega modules

source "/etc/profile.d/00-modulepath.sh"
source "/etc/profile.d/modules.sh"
source "/ceph/hpc/software/cvmfs_env.sh"

module load Python/3.11
module unload OpenSSL/1.1
