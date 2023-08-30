from RemoteRunner import SSHConfig, RemoteRunner, SlurmDirectives

ssh_config = SSHConfig(
    'catalyst.uni-muenster.de',
    60001,
    'username',
    'password'
)
rr = RemoteRunner(ssh_config, '/home/gbircan/remoterunner/')

sd = SlurmDirectives(
    nodes='1',
    mem='1G'
    # partition=SlurmParams.Partition.NORMAL
)

rr.run(
        job_name='remoterunner_testjob',
        slurm_directives=sd,
        container_path='/home/gbircan/singularity/easyChemMLGithub.sif',
        # resource_paths=['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'],
        script_path='./testscript.py',
        # attach=True
    )

# for i in range(0,6):
#     rr.run(
#         slurm_directives=sd,
#         container_path='/home/gbircan/singularity/easyChemMLGithub.sif',
#         # resource_paths=['../_DATASETS/Dreher_and_Doyle_input_data.xlsx'],
#         script_path='./testscript.py',
#         # attach=True
#     )