import cv2
from ultralytics.engine.results import Boxes, Results

from scheduler import Scheduled
from util import crop_box_from_result, get_detection


class ImageSaver(Scheduled):
    def _handle(self, result: Results, box: Boxes, box_path: str, object_path: str, full_id: str):
        _, class_name = get_detection(result, box)
        if not box.id:
            self.logger.warning(f"Box of class {class_name} has no ID")
            return
        image_path = f"{box_path}.jpg"
        crop = crop_box_from_result(result, box)
        cv2.imwrite(image_path, crop)
        self.logger.debug(f"Image saved to {image_path}")