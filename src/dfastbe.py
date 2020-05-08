# coding: utf-8
import logging
import argparse
from pathlib import Path
import dfastbe_kernel


def bank_lines(paramfile, variable):
    """Determines the representative bank lines within the area of interest (First step)."""
    1


def bank_erosion(paramfile, variable, variable2):
    """Determines the expected bank erosion within the area of interest (Second step)."""
    1
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='D-FAST Bank Erosion.')
    parser.add_argument("-i", "--inputfile",
                        default="",
                        required=True,
                        help="name of configuration file",
                        dest="input_file")
    parser.add_argument("-v", "--verbosity",
                        default="INFO",
                        required=False,
                        help="set verbosity level of run-time diagnostics (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
                        dest="verbosity")
    args = parser.parse_args()

    verbosity = args.__dict__["verbosity"].upper()
    verbosity_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if not verbosity in verbosity_levels:
        raise SystemExit('Unknown verbosity level "'+verbosity+'"')

    input_file = args.__dict__["input_file"]
    input_path = Path(input_file).resolve()
    if not input_path.is_file():
        raise SystemExit('Configuration File "'+input_file+'" does not exist!')

    return verbosity, input_file


if __name__ == '__main__':
    verbosity, input_file = parse_arguments()
    logging.basicConfig(level=verbosity,
                        format='%(message)s')
    dfastbe_kernel.main_program(input_file)
