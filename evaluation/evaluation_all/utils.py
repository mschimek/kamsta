from enum import Enum

class Platform(Enum):
    shared_memory = 'shared_memory'
    supermuc = 'supermuc'
    horeka = 'horeka'

    def __str__(self):
        return self.value

def get_num_compute_nodes(platform, num_cores):
    if platform == Platform.supermuc:
        return (num_cores + 47) // 48
    if platform == Platform.horeka:
        return (num_cores + 63) // 64
    raise Exception("inappropriate platform")

def get_queue(platform, num_cores):
    if platform == Platform.supermuc:
        if num_cores <= (16*48):
            return "test"
        elif num_cores <= (768*48):
            return "general"
        else:
            return "large"
    if platform == Platform.horeka:
        if num_cores <= (128):
            return "dev_cpuonly"
        else:
            return "cpuonly"
    raise Exception("inappropriate platform")




