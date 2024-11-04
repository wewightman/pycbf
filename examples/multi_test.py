import multiprocessing as mp
from time import sleep

def on_init(queue):
    global idx
    idx = queue.get()

def f(x):
    global idx
    process = mp.current_process()
    sleep(0.1)
    print(x, idx, process.pid)

nwrkr = 16

manager = mp.Manager()
idQueue = manager.Queue()

for i in range(nwrkr):
    idQueue.put(i)

pool = mp.Pool(nwrkr, initializer=on_init, initargs=(idQueue,))

pool.map(f, range(100))
pool.close()
pool.join()
