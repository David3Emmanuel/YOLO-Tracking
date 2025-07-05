
import queue
import threading
import time
from ultralytics.engine.results import Boxes, Results

from util import get_logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scheduler import Scheduler

class Scheduled:
    def __init__(self):
        self.name = self.__class__.__name__
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        self.scheduler: "Scheduler | None" = None

    def set_scheduler(self, scheduler: "Scheduler"):
        self.scheduler = scheduler
        self.logger = get_logger(self.name, f"{self.scheduler.save_path}/logs")
    
    def worker(self):
        while True:
            item: tuple[Results, Boxes, str, str, str] | None = self.queue.get()
            if item is None:
                break
            result, box, box_path, object_path, full_id = item
            start_time = time.perf_counter()
            try:
                self._handle(result, box, box_path, object_path, full_id)
            except Exception as e:
                self.logger.error(f"Error in {self.name}: {e}")
            finally:
                self.queue.task_done()
                end_time = time.perf_counter()
                self.logger.debug(f"Took {end_time-start_time:.4f} seconds")
    
    def _handle(self, result: Results, box: Boxes, box_path: str, object_path: str, full_id: str):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, result: Results, box: Boxes, box_path: str, object_path: str, full_id: str):
        self.queue.put((result, box, box_path, object_path, full_id))
    
    def cleanup(self):
        self.logger.info(f"Stopping {self.name}...")
        self.queue.put(None)
        self.thread.join()
    
    def __repr__(self):
        return f"<{self.name}>"
