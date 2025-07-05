import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2

from embedding_aggregator import EmbeddingAggregator
from image_saver import ImageSaver
from scheduler import Scheduler
from util import get_logger

classification_model = YOLO(".yolo/models/yolo11n-cls.pt")
model = YOLO(".yolo/models/yolo11n.pt")
layer_indices = [2, 4, 6, 8, 9]


SOURCE = "store.mp4"
SHOULD_PREVIEW = False
SHOULD_SAVE_VIDEO = False
SKIP_FRAMES = 5
OUTPUT_PATH = "output/results"

scheduler = Scheduler(
    save_path=OUTPUT_PATH,
    skip_frames=SKIP_FRAMES,
)\
    | EmbeddingAggregator(classification_model, layer_indices, batch_size=50)\
    # | ImageSaver()\


logger = get_logger(__name__, f"{scheduler.save_path}/logs")
logger.info(f"Tracking {SOURCE} with {model.model_name}")
logger.info(f"Embedding model: {classification_model.model_name}")
logger.info(f"Skipping {SKIP_FRAMES} frames")
logger.info(f"\n{scheduler}")

try:
    results: list[Results] = model.track(
        source=SOURCE, stream=True, verbose=False,
        persist=True, tracker=".yolo/trackers/botsort_with_reid.yaml",
        save=SHOULD_SAVE_VIDEO, project="output",
    )
    start_time = time.perf_counter()

    start_detection_time = time.perf_counter()
    for result in results:
        scheduler(result)
        end_detection_time = time.perf_counter()
        logger.debug(f"Detection took {end_detection_time-start_detection_time:.4f} seconds")
        start_detection_time = end_detection_time

        if SHOULD_PREVIEW:
            im0 = result.plot()
            cv2.imshow("YOLO Tracking", im0)
            if cv2.waitKey(50) & 0xFF == 27: break

except KeyboardInterrupt:
    pass
finally:
    end_time = time.perf_counter()
    scheduler.cleanup()
    cv2.destroyAllWindows()
    logger.info(f"Took {end_time-start_time:.4f} seconds")