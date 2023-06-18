import argparse
from argparse import ArgumentParser

from mir.config import settings
from mir.config.config import Config


def main():
    parser = ArgumentParser(
        prog=settings.PROJECT_PROG,
        description=settings.PROJECT_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'%(prog)s {settings.PROJECT_VERSION}',
    )
    subcommands = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands',
        help='additional help',
        dest='subcommand',
    )
    run_parser = subcommands.add_parser(
        'run',
        help='Run the gRPC server',
    )
    run_parser.add_argument(
        '-c', '--config',
        help='Path to the config yaml file',
    )
    run_parser.add_argument(
        '-p', '--port',
        type=int,
        default=Config.grpc.listen_port,
        help='Port to listen on',
    )
    run_parser.add_argument(
        '-w', '--num-workers',
        type=int,
        default=Config.grpc.num_workers,
        help='Number of workers',
    )
    args = parser.parse_args()
    if args.subcommand == 'run':
        config = Config(config_file=args.config)
        config.grpc.listen_port = args.port
        config.grpc.num_workers = args.num_workers
        if config.config_file is not None:
            from mir.config.reader import read
            config = read(config)
        from mir.server import serve
        serve(config)
    else:
        parser.print_help()
    return None
