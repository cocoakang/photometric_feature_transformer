import psutil
import os

p_list = [p.info for p in psutil.process_iter(attrs=['pid','name','cmdline']) if ('python' in p.info['cmdline'] and 'kill_renderer.py' not in p.info['cmdline'])]
for p in p_list:
    print("kill:"+p['name']+"{}".format(p['pid']))
    os.system('kill {}'.format(p['pid']))
    