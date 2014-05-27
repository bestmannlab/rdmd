from ConfigParser import ConfigParser
import os


CONF_DIR = '../../conf'
config = ConfigParser()
config.read(os.path.join(CONF_DIR, 'config.properties'))

MONITOR=config.get('display','monitor')
SCREEN=int(config.get('display','screen'))
