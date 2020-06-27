import psutil
import os

p_list = [p.info for p in psutil.process_iter(attrs=['pid','name','cmdline']) if '--multiprocessing-fork' in p.info['cmdline']]
for p in p_list:
    print("kill:"+p['name']+"{}".format(p['pid']))
    os.popen('kill {}'.format(p['pid']))