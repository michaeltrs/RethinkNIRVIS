from yaml import load, safe_load, dump
import os


def read_yaml(config_file):
    with open(config_file, 'r') as f:
        try:
            yaml_dict = load(f)
        except:
            yaml_dict = safe_load(f)
    return yaml_dict


def copy_yaml(config_file):
    if type(config_file) is str:
        yfile = read_yaml(config_file)
    elif type(config_file) is dict:
        yfile = config_file
    save_name = os.path.join(yfile['CHECKPOINT']['savedir'], "config_file.yaml")
    i = 2
    while os.path.isfile(save_name):
        save_name = "%s_%d.yaml" % (save_name[:-5], i)
        i += 1
    with open(save_name, 'w') as outfile:
        dump(yfile, outfile, default_flow_style=False)


def get_params_values(args, key, default=None):
    if (key in args) and (args[key] is not None):
        return args[key]
    return default
