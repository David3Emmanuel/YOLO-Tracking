from collections import defaultdict
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from scheduled import Scheduled
from util import crop_box_from_result

class EmbeddingAggregator(Scheduled):
    def __init__(self, classification_model: YOLO, layer_indices: list[int], batch_size: int = 100):
        super().__init__()
        self.classification_model = classification_model
        self.layer_indices = layer_indices
        self.embeddings_aggregate: dict[str, torch.Tensor] = {}
        self.n = defaultdict(int)
        self.batch_size = batch_size
        self.object_paths: dict[str, str] = {}

    def get_representative(self, full_id: str):
        return self.embeddings_aggregate[full_id]/self.n[full_id]

    def _handle(self, result: Results, box: Boxes, box_path: str, object_path: str, full_id: str):
        if full_id not in self.object_paths:
            self.object_paths[full_id] = object_path
        crop = crop_box_from_result(result, box)
        classification_result: torch.Tensor = self.classification_model(
            crop,
            verbose=False,
            embed=self.layer_indices
        )[0]
        self._aggregate(classification_result, full_id)
        
        if self.n[full_id] % self.batch_size == 0:
            self.logger.info(f"Aggregated {self.n[full_id]} embeddings for {full_id}")
            torch.save(self.get_representative(full_id), f"{object_path}/representative.pt")
    
    def _aggregate(self, classification_result: torch.Tensor, full_id: str):
        if full_id in self.embeddings_aggregate:
            self.embeddings_aggregate[full_id] += classification_result
        else:
            self.embeddings_aggregate[full_id] = classification_result.clone()
        self.n[full_id] += 1
    
    def cleanup(self):
        super().cleanup()
        leftovers = 0
        for full_id, object_path in self.object_paths.items():
            leftover = self.n[full_id] % self.batch_size
            if leftover > 0:
                torch.save(self.get_representative(full_id), f"{object_path}/representative.pt")
                self.logger.debug(f"Saved {leftover} leftover embeddings for {full_id}")
                leftovers += leftover
        self.logger.info(f"Saved {leftovers} leftover embeddings")
        self.object_paths.clear()
        self.embeddings_aggregate.clear()
        self.n.clear()