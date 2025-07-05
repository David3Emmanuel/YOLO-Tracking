import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import torch

from consolidator import Consolidator
from embedding_aggregator import EmbeddingAggregator
from image_saver import ImageSaver
from scheduler import Scheduler
from util import get_logger, get_unique_path

import argparse

parser = argparse.ArgumentParser(description='YOLO tracking demo with embedding aggregation')
# Value arguments
parser.add_argument('--source', type=str, default="store.mp4", help='Source video file path')
parser.add_argument('--skip-frames', type=int, default=5, help='Number of frames to skip')
parser.add_argument('--output-path', type=str, default="output/results", help='Output directory path')
# Boolean arguments
parser.add_argument('--preview', action='store_true', help='Show preview window')
parser.add_argument('--save-video', action='store_true', help='Save output video')
parser.add_argument('--save-images', action='store_true', help='Save images')
parser.add_argument('--skip-consolidation', action='store_true', help='Skip consolidation')

args = parser.parse_args()
SOURCE = args.source
SHOULD_PREVIEW = args.preview
SHOULD_SAVE_VIDEO = args.save_video
SKIP_FRAMES = args.skip_frames
OUTPUT_PATH = get_unique_path(args.output_path)
SHOULD_SAVE_IMAGES = args.save_images
SHOULD_CONSOLIDATE = not args.skip_consolidation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classification_model = YOLO(".yolo/models/yolo11n-cls.pt").to(device)
model = YOLO(".yolo/models/yolo11n.pt").to(device)
layer_indices = [2, 4, 6, 8, 9]

# Instantiate and start the consolidator thread before the scheduler
consolidator = Consolidator(OUTPUT_PATH)
consolidator.start()

scheduler = Scheduler(
    save_path=OUTPUT_PATH,
    skip_frames=SKIP_FRAMES,
    consolidator=consolidator,
)\
    | (ImageSaver() if SHOULD_SAVE_IMAGES else None)\
    | EmbeddingAggregator(classification_model, layer_indices, batch_size=50)\


logger = get_logger(__name__, f"{scheduler.save_path}/logs")
logger.info(f"Tracking {SOURCE} with {model.model_name}")
logger.info(f"Embedding model: {classification_model.model_name}")
logger.info(f"Skipping {SKIP_FRAMES} frames")
logger.info(f"\n{scheduler}")

try:
    results: list[Results] = model.track(
        source=SOURCE, stream=True, verbose=False,
        persist=True, tracker="trackers/botsort_with_reid.yaml",
        save=SHOULD_SAVE_VIDEO, project=OUTPUT_PATH,
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
    consolidator.stop()
    consolidator.join()