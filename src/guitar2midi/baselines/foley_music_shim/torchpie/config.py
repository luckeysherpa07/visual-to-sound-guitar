from pyhocon import ConfigFactory

from ._args import args

config = ConfigFactory.parse_file(args.config) if args.config else None
