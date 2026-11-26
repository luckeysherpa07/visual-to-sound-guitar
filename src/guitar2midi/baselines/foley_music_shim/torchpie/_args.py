import argparse

# -c/-e are given by Foley Music's train.py command line; at inference (our wrapper imports core.models, which
# imports torchpie.logging) they are absent: config = None, no experiment folder, log to stdout only.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("-c", "--config", default=None)
_parser.add_argument("-e", "--experiment", default=None)
args, _ = _parser.parse_known_args()
