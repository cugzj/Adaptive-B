import argparse
import client
import config
import logging
import os
import server
import time

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        "ppoavg": server.PPOAvgServer(fl_config), # PPO server disabled
        "ppotrain": server.PPOTrainServer(fl_config), # PPO server disabled
        "favortrain": server.FavorTrainServer(fl_config),
        "favoravg": server.FavorAvgServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global.pth')


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    execution_time = end - start
    print('execution time:{}h'.format(execution_time / 3600))
