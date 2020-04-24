# coding: utf-8
import logging
from configparser import ConfigParser
#from ConfigParser import ConfigParser  # ver. < 3.0


def program_version():
    return 'PRE-ALPHA'


def main_program(input_file):
    # Report program name and version
    program_header()

    # Reading configuration file
    config = read_config(input_file)
    verify_config(config,input_file)

    # Running analysis
    results = run_analysis(config)

    # Writing results
    write_results(results)

    # Finished
    logging.info('Finished')


def program_header():
    logging.critical('D-FAST Bank Erosion '+program_version())
    logging.critical('Copyright (c) 2020 Deltares.')
    logging.critical('')
    logging.critical('This program is distributed under the terms of the')
    logging.critical('GNU Lesser General Public License Version 2.1; see')
    logging.critical('the LICENSE.md file for details.')
    logging.critical('')
    logging.info('Source code location:')
    logging.info('https://github.com/Deltares/D-FAST_Bank_Erosion')
    logging.info('')


def read_config(input_file):
    logging.info('Reading configuration file '+input_file)

    # instantiate file parser
    config = ConfigParser()

    # open the configuration file
    fid = open(input_file,'r')

    # read and parse the configuration file
    config.read_file(fid)

    # close the configuration file
    fid.close()
    return config


def logkeyvalue(level,key,val):
    logging.log(level,str.format('%-30s: %s' % (key,val)))


def verify_config(config,input_file):
    logging.info('Verifying configuration file')
    try:
        filename = config['General']['Discharge']
        logkeyvalue(logging.DEBUG,'Discharges',filename)
    except:
        raise SystemExit('Unable to read General\Discharge from "'+input_file+'"!')
    try:
        filename = config['General']['FileX']
        logkeyvalue(logging.DEBUG,'Input file',filename)
    except:
        raise SystemExit('Unable to read General\FileX from "'+input_file+'"!')
    logging.debug('')
    return


def read_dflowfm(map_file):
    logging.debug('Loading file "'+map_file+'"')
    data = 1
    return data


def run_analysis(config):
    # Load data
    logging.info('Loading data')
    data1 = read_dflowfm(config['General']['Discharge'])
    data2 = read_dflowfm(config['General']['FileX'])
    logging.debug('')

    # Do actual analysis
    logging.info('Running analysis')
    logging.debug('')
    return 0


def write_results(results):
    logging.info('Writing results')
    logging.debug('')
