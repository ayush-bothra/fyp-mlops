import numpy as np


class CoresetBuffer:
    def __init__(self, capacity=200, window_size=100, k=1.5):
        self.capacity = capacity
        self.window_size = window_size
        self.k = k
        self.embeddings = []
        self.items = []
        self.score_window = []

    def nearest_neighbor_distance(self, embedding):
        normalized = embedding / np.linalg.norm(embedding)
        if not self.embeddings:
            return 1.0
        similarities = np.array(self.embeddings) @ normalized
        return 1.0 - np.max(similarities)

    def admission_threshold(self):
        if len(self.score_window) < 4:
            return -np.inf
        q1, q2, q3 = np.percentile(self.score_window, [25, 50, 75])
        sqir = (q3 - q1) / 2.0
        return q2 + self.k * sqir

    def evict_most_redundant(self):
        matrix = np.array(self.embeddings)
        similarity_matrix = matrix @ matrix.T
        np.fill_diagonal(similarity_matrix, -np.inf)
        evict_index = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)[0]
        self.embeddings.pop(evict_index)
        self.items.pop(evict_index)

    def evaluate_and_add(self, embedding, item):
        score = self.nearest_neighbor_distance(embedding)
        self.score_window.append(score)
        if len(self.score_window) > self.window_size:
            self.score_window.pop(0)

        threshold = self.admission_threshold()
        admitted = score > threshold
        if admitted:
            normalized = embedding / np.linalg.norm(embedding)
            self.embeddings.append(normalized)
            self.items.append(item)
            if len(self.embeddings) > self.capacity:
                self.evict_most_redundant()
        return admitted, score
