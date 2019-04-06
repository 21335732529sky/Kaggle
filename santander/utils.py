from threading import Thread, get_ident
from multiprocessing import Process, Queue
from queue import Empty
from abc import abstractmethod
import copy

class ParallelExecutorBase:
  def __init__(self):
    self.queue = Queue()

  @abstractmethod
  def work(self, item):
    raise NotImplementedError()

  @abstractmethod
  def make_items(self, **kwargs):
    raise NotImplementedError()

  def worker(self):
    while True:
      try:
        item = self.queue.get(timeout=1)
      except Empty:
        break
      self.work(item)

  def run(self, num_worker, **kwargs):
    [self.queue.put(e) for e in self.make_items(**kwargs)]
    processes = []
    [processes.append(Process(target=self.worker)) for _ in range(num_worker)]
    [p.start() for p in processes]

    [p.join() for p in processes]