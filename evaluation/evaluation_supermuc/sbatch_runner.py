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
        # print(yaml.dump(data))
        # print(data["name"])
        # print(data["graphs"])
        self.name = data["name"]
        self.graphs = data["graphs"]
        self.list_num_cores = data["cores"]
        self.list_num_threads = data["threads"]
        self.time_limit = data["time_limit"]
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
                        # print(experiment_config)
                        experiments[num_cores] += [Experiment(self.name, num_cores, num_threads, experiment_config)]
                        # print("list:")
                        # for entry in experiments[num_cores]:
                        #     print(entry)
        return experiments


def get_queue(num_cores):
    if num_cores <= (16*48):
        return "test"
    elif num_cores <= (768*48):
        return "general"
    else:
        return "large"


def get_num_compute_nodes(num_cores):
    return (num_cores + 47) // 48


def build_executable(path_to_exec, **kwargs):
    executable = Path(os.getcwd()) / "benchmarks" / "mst_benchmarks"
    command = [str(path_to_exec)]
    for arg, value in kwargs.items():
        command.append("--" + arg)
        if not isinstance(value, bool):
            command.append(str(value))
    return ' '.join(command)

class Jobfile:
    def __init__(self, name, output_directory, num_cores, jobfile_output_directory, path_to_exec, path_supermuc_load_config, time_limit):
        self.output_directory = Path(output_directory)
        self.jobfile_output_directory = Path(jobfile_output_directory)
        self.time_limit = time_limit
        self.script_path = Path(os.path.dirname(__file__))
        self.path_to_exec = path_to_exec
        self.output_directory = self.output_directory / str(num_cores)
        self.output_directory.mkdir(exist_ok=True, parents=True)

        jobfile_subs = {}
        jobfile_subs["nodes"] = get_num_compute_nodes(num_cores)
        jobfile_subs["time"] = time_limit
        self.job_queue = get_queue(num_cores)
        jobfile_subs["job_queue"] = self.job_queue
        jobfile_subs["num_switches"] = 1 if jobfile_subs["job_queue"] != "large" else 2
        jobfile_subs["output_log"] = str(self.output_directory / "log.txt")
        jobfile_subs["error_log"] = str(self.output_directory / "err_log.txt")
        jobfile_subs["supermuc_load_config"] = path_supermuc_load_config
        jobname = name + "_" + str(num_cores)
        jobfile_subs["job_name"] = jobname
        jobfile_subs["supermuc_account"] = os.getenv('SUPERMUC_ACCOUNT')
        jobfile_subs["mail_address"] = os.getenv('MAIL_ADDRESS')

        with open(self.script_path /
                  "jobfile_template.txt") as jobfile_template_file:
            jobfile_template = jobfile_template_file.read()
        jobfile_template = Template(jobfile_template)
        jobfile = jobfile_template.substitute(jobfile_subs)
        self.jobfile_path = self.jobfile_output_directory / jobname
        with open(self.jobfile_path, "w") as job:
            job.write(jobfile)

    def add_to_jobfile(self, experiment):
        with open(self.script_path /
                  "command_template.txt") as command_template_file:
            command_template = command_template_file.read()
        command_template = Template(command_template)
        command_subs = {}
        print(experiment)
        # command_subs["jobname"]
        command_subs["num_threads"] = experiment.num_threads
        command_subs["num_mpi_procs"] = experiment.num_cores // experiment.num_threads
        command_subs["num_cpus_per_mpi_proc"] = experiment.num_threads
        command_subs["num_mpi_procs_per_node"] = 48 // experiment.num_threads
        command_subs["num_mpi_procs_per_node"] = 48 // experiment.num_threads
        command_subs["sleep_duration"] = 20 if self.job_queue == "large" else 10
        experiment.params["outfile"] = self.output_directory / heuristic_unique_name_gen(experiment)
        # TODO change in exec
        experiment.params["threads"] = experiment.num_threads
        command_subs["jobname"] = experiment.params["outfile"]
        command_subs["cmd"] = build_executable(self.path_to_exec, **experiment.params)
        command = command_template.safe_substitute(command_subs)
        with open(self.jobfile_path, "a") as job:
            job.write(command)
