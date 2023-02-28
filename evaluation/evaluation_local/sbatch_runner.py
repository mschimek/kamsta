from pathlib import Path
import yaml
import os
from string import Template


def heuristic_unique_name_gen(experiment):
    params = experiment.params
    unique_name = ""
    unique_name += "threads_" + str(experiment.num_threads) + "_"
    key = "graphtype"
    if(key in params):
        unique_name += key + "_" + str(params[key]) + "_"
    key = "algorithm"
    if(key in params):
        unique_name += key + "_" + str(params[key]) + "_"
    key = "log_n"
    if(key in params):
        unique_name += key + "_" + str(params[key]) + "_"
    key = "log_m"
    if(key in params):
        unique_name += key + "_" + str(params[key]) + "_"
    key = "local_kernelization"
    if(key in params):
        unique_name += key + "_" + str(params[key]) + "_"
    unique_name += "id_" + str(heuristic_unique_name_gen.counter) + ".txt"
    heuristic_unique_name_gen.counter += 1
    return unique_name
heuristic_unique_name_gen.counter = 0


class Experiment:
    def __init__(self, name, num_cores, num_threads, params):
        self.name = name + "_p" + str(num_cores) + "_t" + str(num_threads)
        self.num_cores = num_cores
        self.num_threads = num_threads
        self.params = params

    def __str__(self):
        desc = "#cores: " + str(self.num_cores) + " #threads: " + str(self.num_threads)
        for param, value in self.params.items():
            desc = desc + "\n\t --" + str(param) + " " + str(value)
        return desc


def explode_combinatorically(config):
    experiments = []
    for param, values in config.items():
        if type(values) == list:
            for value in values:
                experiment = config.copy()
                experiment[param] = value
                tmp = explode_combinatorically(experiment)
                experiments = experiments + tmp
            break
    if not experiments:
        return [config]
    return experiments


class ExperimentSuite:
    def __init__(self, path):
        self.load_yaml_experiment_config(path)

    def load_yaml_experiment_config(self, path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        self.name = data["name"]
        self.graphs = data["graphs"]
        self.list_num_cores = data["cores"]
        self.list_num_threads = data["threads"]
        self.config = data["config"]

    def to_single_experiments(self):
        experiment_configs = explode_combinatorically(self.config)
        experiments = {}
        print(self.config, "\n")
        for num_cores in self.list_num_cores:
            experiments[num_cores] = []
            for num_threads in self.list_num_threads:
                for experiment_config in experiment_configs:
                    for graph in self.graphs:
                        experiment_config = experiment_config.copy()
                        experiment_config.update(graph)
                        experiments[num_cores] += [Experiment(self.name, num_cores, num_threads, experiment_config)]
        return experiments

def build_executable(path_to_exec, **kwargs):
    executable = Path(os.getcwd()) / "benchmarks" / "mst_benchmarks"
    command = [str(path_to_exec)]
    for arg, value in kwargs.items():
        command.append("--" + arg)
        if not isinstance(value, bool):
            command.append(str(value))
    return ' '.join(command)

class Jobfile:
    def __init__(self, name, output_directory, num_cores, jobfile_output_directory, path_to_exec):
        self.output_directory = Path(output_directory)
        self.jobfile_output_directory = Path(jobfile_output_directory)
        self.script_path = Path(os.path.dirname(__file__))
        self.path_to_exec = path_to_exec
        self.output_directory = self.output_directory / str(num_cores)
        self.output_directory.mkdir(exist_ok=True, parents=True)

        jobname = name + "_" + str(num_cores)
        self.jobfile_path = self.jobfile_output_directory / jobname

    def add_to_jobfile(self, experiment):
        with open(self.script_path /
                  "command_template.txt") as command_template_file:
            command_template = command_template_file.read()
        command_template = Template(command_template)
        command_subs = {}
        print(experiment)
        command_subs["num_threads"] = experiment.num_threads
        command_subs["num_mpi_procs"] = experiment.num_cores // experiment.num_threads
        command_subs["num_cpus_per_mpi_proc"] = experiment.num_threads
        command_subs["sleep_duration"] = 2
        command_subs["output_log"] = str(self.output_directory / "log.txt")
        command_subs["error_log"] = str(self.output_directory / "err_log.txt")
        experiment.params["threads"] = experiment.num_threads
        command_subs["cmd"] = build_executable(self.path_to_exec, **experiment.params)
        command = command_template.safe_substitute(command_subs)
        with open(self.jobfile_path, "a") as job:
            job.write(command)
