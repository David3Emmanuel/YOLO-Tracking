import time
from util import get_logger
import os
import shutil
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt

class Consolidator:
    def __init__(self, embeddings_root: str, similarity_threshold: float = 0.9):
        self.embeddings_root = embeddings_root
        self.similarity_threshold = similarity_threshold
        self.consolidation_results = {}
        self.logger = get_logger(self.__class__.__name__, f"{self.embeddings_root}/logs")
    
    def get_representative_id(self, class_name, object_id):
        return self.consolidation_results[class_name][object_id]
    
    def _rearrange_files(self):
        for class_name, results in self.consolidation_results.items():
            for object_id, representative_id in results.items():
                if object_id == representative_id: continue
                old_folder = os.path.join(self.embeddings_root, f"{class_name}#{object_id}")
                new_folder = os.path.join(self.embeddings_root, f"{class_name}#{representative_id}")
                old_file = os.path.join(old_folder, "representative.pt")
                new_file = os.path.join(new_folder, f"representative_{object_id}.pt")
                shutil.copy(old_file, new_file)
                shutil.copytree(old_folder, new_folder, dirs_exist_ok=True)
                shutil.rmtree(old_folder)
                self.logger.debug(f"Moved {old_folder} to {new_folder}")
        self.logger.info("Rearranged files")

    def _find_representative_files(self):
        """Recursively find all representative.pt files and group by class."""
        class_to_files = defaultdict(list)
        
        # Assume path: <self.embeddings_root>/<class_name>#<object_id>/representative.pt
        for folder in os.listdir(self.embeddings_root):
            folder_path = os.path.join(self.embeddings_root, folder)
            if os.path.isdir(folder_path):
                if "representative.pt" in os.listdir(folder_path):
                    class_name = folder.split("#")[0]
                    object_id = int(folder.split("#")[1])
                    class_to_files[class_name].append((object_id, os.path.join(folder_path, "representative.pt")))
        
        return class_to_files

    def consolidate(self, rearrange: bool = True, visualize: bool = False):
        start_time = time.perf_counter()
        """For each class, group object ids with similar embeddings."""
        class_to_files = self._find_representative_files()
        
        for class_name, obj_files in class_to_files.items():
            embeddings, object_ids = self._load_embeddings(obj_files)
            sim_matrix = self._compute_similarity(embeddings)
            if visualize:
                self.visualize_similarity(class_name, sim_matrix, object_ids)
            results = self._consolidate_class(object_ids, sim_matrix)
            self.consolidation_results[class_name] = results
            self.logger.debug(f"Consolidated {class_name} with {len(results)} objects")
        if rearrange:
            self._rearrange_files()
        end_time = time.perf_counter()
        if not visualize:
            self.logger.info(f"Took {end_time-start_time:.4f} seconds")

    def visualize_similarity(self, class_name, sim_matrix: torch.Tensor, object_ids: list[int]):
        sim_matrix = sim_matrix.clone().cpu()
        sim_matrix = torch.triu(sim_matrix)
        
        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='white')
        plt.title(f"Similarity Matrix for {class_name}")
        
        plt.imshow(sim_matrix, cmap=cmap, vmin=0.5, vmax=1)
        plt.xticks(range(len(object_ids)), [str(id) for id in object_ids], rotation=90)
        plt.yticks(range(len(object_ids)), [str(id) for id in object_ids])
        plt.colorbar()
        plt.savefig(f"{self.embeddings_root}/similarity_matrix_{class_name}.png")
        plt.close()

    def _load_embeddings(self, obj_files):
        embeddings = []
        object_ids = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for object_id, file_path in obj_files:
            emb = torch.load(file_path, map_location=device)
            embeddings.append(emb)
            object_ids.append(object_id)
        self.logger.debug(f"Loaded {len(embeddings)} embeddings for {len(object_ids)} objects")
        return embeddings, object_ids
    
    def _compute_similarity(self, embeddings):
        embeddings = torch.stack(embeddings)
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        return sim_matrix

    def _consolidate_class(self, object_ids, sim_matrix):
        # TODO: This is a naive implementation chosen for speed.
        groups: dict[int, set[int]] = defaultdict(set)
        
        for i in range(len(object_ids)):
            for j in range(i, len(object_ids)):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    groups[object_ids[i]].add(object_ids[j])
                    groups[object_ids[j]].add(object_ids[i])
        
        return {
            id: min(group)
            for id, group in groups.items()
        }
    
if __name__ == "__main__":
    consolidator = Consolidator("output/results")
    consolidator.consolidate(rearrange=True, visualize=True)