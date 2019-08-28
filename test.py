import threading
import pdb
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def next(self):
        self.lock.acquire()
        try:
            return self.it.next()
        finally:
            self.lock.release()

gen = (x*2 for x in [1,2,3,4])
pdb.set_trace()
g2 = LockedIterator(gen)
print list(g2)
