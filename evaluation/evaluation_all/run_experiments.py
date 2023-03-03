import argparse
import shutil
import os
import stat
from sbatch_runner import ExperimentSuite, Jobfile
from utils import Platform
from pathlib import Path
from datetime import date


# generates a jobfile for each number of cores
def generate_jobfiles(path_to_suite, jobfile_dir, output_dir, path_to_exec, path_load_config, platform):
    suite = ExperimentSuite(path_to_suite, platform)
    experiments = suite.to_single_experiments()
    for num_cores, experiments in experiments.items():
        num_threads = experiments[0].num_threads # will only be used for horeka as this information is needed in the jobfile here
        jobfile = Jobfile(platform, suite.name, output_dir, num_cores, num_threads, jobfile_dir, path_to_exec, path_load_config, suite.time_limit)
        for experiment in experiments:
            jobfile.add_to_jobfile(experiment)

def generate_path_to_config_file(platform):
    if(platform == Platform.supermuc):
        return Path(os.getcwd()) / "load_configs" / "supermuc.sh"
    return Path(os.getcwd()) / "load_configs" / "empty.sh"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_parent_directory', required=True, help="path to directory where the experiment result subdirectory will be created.")
    parser.add_argument('--experiment_name', required=True, help="name of the experiment result subdirectory.")
    parser.add_argument('--experiment_suite', required=True, help="experiment suite that will be executed in this experiment.")
    parser.add_argument('--platform', type=Platform, choices=list(Platform), required=True)

    args = parser.parse_args()
    print(args.experiment_parent_directory)
    print(args.experiment_name)
    print(args.experiment_suite)
    print(args.platform)
    parser = argparse.ArgumentParser()
    path_to_suite = Path(os.getcwd()) / args.experiment_suite
    path_to_load_config = generate_path_to_config_file(args.platform)
    parent_dir = Path(os.path.expanduser(args.experiment_parent_directory))

    today = date.today()
    augmented_experiment_name = args.experiment_name + "_" + today.strftime("%d_%m_%Y")
    experiment_dir = parent_dir / augmented_experiment_name
    experiment_dir.mkdir(exist_ok=True, parents=True)
    path_to_exec = Path(os.getcwd()) / "../../build/benchmarks/mst_benchmarks"
    path_to_exec_in_exp_dir = experiment_dir / "mst_benchmarks"
    path_to_submitted_jobs = experiment_dir / "submitted_jobs"
    path_to_submitted_jobs.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path_to_exec, path_to_exec_in_exp_dir)
    st = os.stat(path_to_exec)
    os.chmod(path_to_exec_in_exp_dir, st.st_mode | stat.S_IEXEC)

    shutil.copyfile(path_to_suite, experiment_dir / "suite.yaml")
    load_config_in_exp_dir = experiment_dir / "load_config.sh"
    shutil.copyfile(path_to_load_config, load_config_in_exp_dir)

    jobfile_dir = experiment_dir / "jobfiles"
    jobfile_dir.mkdir(exist_ok=True, parents=True)
    output_dir = experiment_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    generate_jobfiles(path_to_suite, jobfile_dir, output_dir, path_to_exec_in_exp_dir, load_config_in_exp_dir, args.platform)


if __name__ == "__main__":
    main()
