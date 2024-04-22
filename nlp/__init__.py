# -*- coding: utf-8 -*-

"""Top-level package for nlp."""

__author__ = """A Student"""
__email__ = 'student@example.com'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.nlp/nlp.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
# here is an example.
def write_default_config(path):
    with open(path, 'wt') as w:
        w.write('[data]\n')
        w.write('url = https://www.dropbox.com/scl/fi/8afm3cbr1ui1j3qrtv1u9/train.csv?rlkey=d0y73zduv1ira37d5xyd0sg2m&st=tfkqctcq&dl=1\n')  # Corrected URL
        w.write('file = %s%s%s\n' % (os.path.dirname(path), os.path.sep, 'nli.csv'))  # Corrected 'file' option


# Find NLP_HOME path
if 'NLP_HOME' in os.environ:
    nlp_path = os.environ['NLP_HOME']
else:
    nlp_path = os.environ['HOME'] + os.path.sep + '.nlp' + os.path.sep

# Make nlp directory if not present
try:
    os.makedirs(nlp_path)
except:
    pass

# main config file.
config_path = nlp_path + 'nlp.cfg'
# classifier
clf_path = nlp_path + 'clf.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)