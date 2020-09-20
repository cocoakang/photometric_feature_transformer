import numpy as np
import random
import threading

def run(name,seed=None,produce=True):
    if seed is not None:
        np.random.seed(seed)
    if produce:
        print("[THREAD] {} says:{}".format(name,np.random.randint(0,10)))
    else:
        print("[THREAD] {} sets seed".format(name))

class Mine_Pro():
    def __init__(self,name,seed,produce):
        self.name = name
        self.seed = seed
        self.produce = produce
    
    def start(self):
        self.generator = threading.Thread(target=run, args=(
            self.name,
            self.seed,
            self.produce
        ))
        self.generator.setDaemon(True)
        self.generator.start()


seed_main = 666
seed_process_1 = 999
seed_process_2 = 233

run("seed_main",seed_main)
run("seed_process_1",seed_process_1)
run("seed_process_2",seed_process_2)
# run("seed_main",seed_main,False)
# print("np.random.randint(0,10):",np.random.randint(0,10))

run("seed_process_2",seed_process_2,False)
test_mine = Mine_Pro("seed_process_1",seed_process_1,False)
test_mine.start()
print("np.random.randint(0,10):",np.random.randint(0,10))
