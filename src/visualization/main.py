import os
import sys
from yaml import load as yaml_load
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from src.data.make_dataset import LoadDataframe
from src.visualization.visualize import Visu
from tornado.ioloop import IOLoop
from bokeh.server.server import Server

def _load_config_file(config_file):
    """
    Load configuration file
    :param config_file: is the configuration file
    :return: configuration
    :rtype: dict
    """
    with open(config_file) as yml_config:
        return yaml_load(yml_config)

def _build_configuration(config_file):
    """
    Build the operation configuration dict
    :param config_file: is the path to the yaml config_file
    :type: string
    :return: config: global configuration
    :rtype dict
    """
    # yaml config
    config = _load_config_file(config_file)
    return config


def main(config_file='/home/ml/Documents/crimes_chigaco/config/config.yml'):
    """
    Script entry point function
    """
    # Check if configuration file parameter is set and if the file exists
    if not config_file:
        print('Configuration file is mandatory, abort')
        sys.exit(1)
    elif not os.path.isfile(config_file):
        print('"{}" configuration file not exists, abort'.format(config_file))
        sys.exit(1)

    # Load configuration from YAML file
    # build configuration
    config = _build_configuration(config_file)
    obj_df_loaded = LoadDataframe(config, '2001', '2017')
    df_merged = obj_df_loaded.df_merged()
    obj_visu = Visu(config, df_merged)
    io_loop = IOLoop.current()
    bokeh_app = Application(FunctionHandler(obj_visu.modify_doc))
    server = Server({"/": bokeh_app}, io_loop=io_loop)
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")
    io_loop.add_callback(server.show, "/")
    io_loop.start()

if __name__ == "__main__":
    if len(sys.argv) != 1:
        main(sys.argv[1])
    else:
        main()
