import argparse
import shutil
import os
import stat
from sbatch_runner import ExperimentSuite, Jobfile
from pathlib import Path
from datetime import date


# generates a jobfile for each number of cores
def generate_jobfiles(path_to_suite, jobfile_dir, output_dir, path_to_exec):
    suite = ExperimentSuite(path_to_suite)
    experiments = suite.to_single_experiments()
    for num_cores, experiments in experiments.items():
        num_threads = experiments[0].num_threads
        jobfile = Jobfile(suite.name, output_dir, num_cores, num_threads, jobfile_dir, path_to_exec, suite.time_limit)
        for experiment in experiments:
            jobfile.add_to_jobfile(experiment)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_parent_directory', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--experiment_suite', required=True)

    args = parser.parse_args()
    print(args.experiment_parent_directory)
    print(args.experiment_name)
    print(args.experiment_suite)
    parser = argparse.ArgumentParser()
    path_to_suite = Path(os.getcwd()) / args.experiment_suite
    parent_dir = Path(os.path.expanduser(args.experiment_parent_directory))

    today = date.today()
    augmented_experiment_name = args.experiment_name + "_" + today.strftime("%d_%m_%Y")
    experiment_dir = parent_dir / augmented_experiment_name
    experiment_dir.mkdir(exist_ok=True, parents=True)
    path_to_exec = Path(os.getcwd()) / "../build/benchmarks/mst_benchmarks"
    path_to_exec_in_exp_dir = experiment_dir / "mst_benchmarks"
    path_to_submitted_jobs = experiment_dir / "submitted_jobs"
    path_to_submitted_jobs.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path_to_exec, path_to_exec_in_exp_dir)
    st = os.stat(path_to_exec)
    os.chmod(path_to_exec_in_exp_dir, st.st_mode | stat.S_IEXEC)

    shutil.copyfile(path_to_suite, experiment_dir / "suite.yaml")

    jobfile_dir = experiment_dir / "jobfiles"
    jobfile_dir.mkdir(exist_ok=True, parents=True)
    output_dir = experiment_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    generate_jobfiles(path_to_suite, jobfile_dir, output_dir, path_to_exec_in_exp_dir)


if __name__ == "__main__":
    main()
