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


class YoloTracker:
    def __init__(self, source, skip_frames, output_path, preview, save_video, save_images, skip_consolidation):
        self.SOURCE = source
        self.SHOULD_PREVIEW = preview
        self.SHOULD_SAVE_VIDEO = save_video
        self.SKIP_FRAMES = skip_frames
        self.OUTPUT_PATH = get_unique_path(output_path)
        self.SHOULD_SAVE_IMAGES = save_images
        self.SHOULD_CONSOLIDATE = not skip_consolidation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.classification_model = YOLO(".yolo/models/yolo11n-cls.pt").to(self.device)
        self.model = YOLO(".yolo/models/yolo11n.pt").to(self.device)
        self.layer_indices = [2, 4, 6, 8, 9]

        # Instantiate and start the consolidator thread before the scheduler
        self.consolidator = Consolidator(self.OUTPUT_PATH)
        self.consolidator.start()

        self.scheduler = Scheduler(
            save_path=self.OUTPUT_PATH,
            skip_frames=self.SKIP_FRAMES,
            consolidator=self.consolidator,
        )\
            | (ImageSaver() if self.SHOULD_SAVE_IMAGES else None)\
            | EmbeddingAggregator(self.classification_model, self.layer_indices, batch_size=50)

        self.logger = get_logger(__name__, f"{self.scheduler.save_path}/logs")
        self.logger.info(f"Tracking {self.SOURCE} with {self.model.model_name}")
        self.logger.info(f"Embedding model: {self.classification_model.model_name}")
        self.logger.info(f"Skipping {self.SKIP_FRAMES} frames")
        self.logger.info(f"\n{self.scheduler}")

        self._start_time = None

    def run(self):
        results: list[Results] = self.model.track(
            source=self.SOURCE, stream=True, verbose=False,
            persist=True, tracker="trackers/botsort_with_reid.yaml",
            save=self.SHOULD_SAVE_VIDEO, project=self.OUTPUT_PATH,
        )
        self._start_time = time.perf_counter()
        start_detection_time = time.perf_counter()
        for result in results:
            self.scheduler(result)
            end_detection_time = time.perf_counter()
            self.logger.debug(f"Detection took {end_detection_time-start_detection_time:.4f} seconds")
            start_detection_time = end_detection_time

            if self.SHOULD_PREVIEW:
                im0 = result.plot()
                cv2.imshow("YOLO Tracking", im0)
                if cv2.waitKey(50) & 0xFF == 27:
                    break

    def cleanup(self):
        end_time = time.perf_counter()
        self.scheduler.cleanup()
        cv2.destroyAllWindows()
        if self._start_time is not None:
            self.logger.info(f"Took {end_time-self._start_time:.4f} seconds")
        self.consolidator.cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO tracking demo with embedding aggregation')
    parser.add_argument('--source', type=str, default="store.mp4", help='Source video file path')
    parser.add_argument('--skip-frames', type=int, default=5, help='Number of frames to skip')
    parser.add_argument('--output-path', type=str, default="output/results", help='Output directory path')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--save-images', action='store_true', help='Save images')
    parser.add_argument('--skip-consolidation', action='store_true', help='Skip consolidation')
    args = parser.parse_args()
    tracker = YoloTracker(
        source=args.source,
        skip_frames=args.skip_frames,
        output_path=args.output_path,
        preview=args.preview,
        save_video=args.save_video,
        save_images=args.save_images,
        skip_consolidation=args.skip_consolidation
    )
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.cleanup()