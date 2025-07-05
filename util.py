import sys, os
from ultralytics.engine.results import Results, Boxes
import logging

def get_logger(name: str, log_path: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        print("Logger already has handlers", file=sys.stderr)
        return logger
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s::%(levelname)s %(message)s",
        "%H:%M:%S"
    )

    os.makedirs(log_path, exist_ok=True)
    
    file_handler = logging.FileHandler(f"{log_path}/{name}.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def crop_box_from_result(result: Results, box: Boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    crop = result.orig_img[y1:y2, x1:x2].copy()
    return crop

def get_classification(results: list[Results]):
    result = results[0]
    class_id = result.probs.top1
    class_name = result.names[class_id]
    return class_id, class_name

def get_detection(result: Results, box: Boxes):
    class_id = int(box.cls.item())
    class_name = result.names[class_id]
    return class_id, class_name