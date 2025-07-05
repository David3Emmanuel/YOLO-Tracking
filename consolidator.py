import os
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt

class Consolidator:
    def __init__(self, embeddings_root: str, similarity_threshold: float = 0.9):
        self.embeddings_root = embeddings_root
        self.similarity_threshold = similarity_threshold

    def _find_representative_files(self):
        """Recursively find all representative.pt files and group by class."""
        class_to_files = defaultdict(list)
        for root, dirs, files in os.walk(self.embeddings_root):
            for file in files:
                if file == "representative.pt":
                    # Assume path: .../<class_name>#<object_id>/representative.pt
                    folder_name = os.path.basename(root)
                    class_name = folder_name.split("#")[0]
                    object_id = folder_name.split("#")[1]
                    class_to_files[class_name].append((object_id, os.path.join(root, file)))
        return class_to_files

    def consolidate(self):
        """For each class, group object ids with similar embeddings."""
        class_to_files = self._find_representative_files()
        
        for class_name, obj_files in class_to_files.items():
            embeddings, object_ids = self._load_embeddings(obj_files)
            sim_matrix = self._compute_similarity(embeddings)
            # self._consolidate_class(class_name, object_ids, sim_matrix)
            self.visualize_similarity(class_name, sim_matrix, object_ids)

    def visualize_similarity(self, class_name, sim_matrix, object_ids):
        plt.title(f"Similarity Matrix for {class_name}")
        plt.imshow(sim_matrix, cmap='viridis')
        plt.xticks(range(len(object_ids)), object_ids, rotation=90)
        plt.yticks(range(len(object_ids)), object_ids)
        plt.colorbar()
        plt.show()

    def _load_embeddings(self, obj_files):
        embeddings = []
        object_ids = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for object_id, file_path in obj_files:
            emb = torch.load(file_path, map_location=device)
            embeddings.append(emb)
            object_ids.append(object_id)
        return embeddings, object_ids
    
    def _compute_similarity(self, embeddings):
        embeddings = torch.stack(embeddings)
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        return sim_matrix

    def _consolidate_class(self, class_name, object_ids, sim_matrix):
        n = len(object_ids)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        for i in range(n):
            for j in range(i+1, n):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    union(i, j)
        # Build groups
        groups = defaultdict(list)
        for idx, oid in enumerate(object_ids):
            root = find(idx)
            groups[root].append(oid)
        return groups
    
if __name__ == "__main__":
    consolidator = Consolidator("output/results")
    consolidator.consolidate()