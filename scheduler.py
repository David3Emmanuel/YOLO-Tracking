import os
import queue
import threading
import time
from ultralytics.engine.results import Results

from util import get_detection, get_logger
from scheduled import Scheduled

class Scheduler:
    def __init__(self, skip_frames: int, save_path: str):
        self.save_path = self._get_unique_path(save_path)
        self.logger = get_logger(self.__class__.__name__, f"{self.save_path}/logs")
        self.logger.info(f"Storing output in {self.save_path}")
        
        self.skip_frames = skip_frames
        self.to_skip = 0
        self.pipes: list[Scheduled] = []
        
        self.schedule_queue = queue.Queue()
        self.thread = threading.Thread(target=self.schedule_worker, daemon=True)
        self.thread.start()
    
    def schedule_worker(self):
        while True:
            item: Results | None = self.schedule_queue.get()
            if item is None:
                break
            result = item
            if self.to_skip > 0:
                self.to_skip -= 1
                continue
            self.to_skip = self.skip_frames
            
            start_time = time.perf_counter()
            try:
                if not result.boxes:
                    self.logger.warning("No boxes found in result")
                    continue
                
                for box in result.boxes:
                    _, class_name = get_detection(result, box)
                    if not box.id:
                        self.logger.warning(f"Box of class {class_name} has no ID")
                        continue
                    box_id = int(box.id.item())
                    self.logger.debug(f"Handling box {box_id} of class {class_name}")
                    full_id = f"{class_name}#{box_id}"
                    object_path = f"{self.save_path}/{full_id}"
                    os.makedirs(object_path, exist_ok=True)
                    box_path = f"{object_path}/{time.time()}"
                    
                    for pipe in self.pipes:
                        pipe(result, box, box_path, object_path, full_id)
                
            except Exception as e:
                self.logger.error(f"Error in schedule_worker: {e}")
            finally:
                self.schedule_queue.task_done()
                end_time = time.perf_counter()
                self.logger.debug(f"Took {end_time-start_time:.4f} seconds")
    
    def _get_unique_path(self, save_path: str):
        i = 2
        _save_path = save_path
        while os.path.exists(_save_path):
            _save_path = f"{save_path}_{i}"
            i += 1
        os.makedirs(_save_path, exist_ok=True)
        return _save_path
    
    
    def __or__(self, pipe: Scheduled | None):
        if pipe is None:
            return self
        self.pipes.append(pipe)
        pipe.set_scheduler(self)
        return self
    
    def __call__(self, result: Results):
        self.schedule_queue.put(result)
    
    def cleanup(self):
        self.logger.info("Stopping scheduler...")
        self.schedule_queue.put(None)
        self.thread.join()
        
        for pipe in self.pipes:
            pipe.cleanup()
    
    def __repr__(self):
        out = f"<{self.__class__.__name__}>"
        for i, pipe in enumerate(self.pipes):
            out += f"\n  {i}: {pipe}"
        return out
