import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buffer.selector import UnsupervisedSigLIPBuffer


def make_synthetic_cluster_points(direction, count, noise_scale, embedding_dim, rng):
    base = np.zeros(embedding_dim)
    base[direction] = 1.0
    return base + rng.normal(scale=noise_scale, size=(count, embedding_dim))


def test_kmeans_assigns_synthetic_clusters_correctly():
    rng = np.random.default_rng(0)
    embedding_dim = 8
    buffer = UnsupervisedSigLIPBuffer(n_clusters=3, embedding_dim=embedding_dim, window_size=50)

    cluster_a = make_synthetic_cluster_points(0, 30, 0.02, embedding_dim, rng)
    cluster_b = make_synthetic_cluster_points(3, 30, 0.02, embedding_dim, rng)
    cluster_c = make_synthetic_cluster_points(6, 30, 0.02, embedding_dim, rng)

    for point in np.concatenate([cluster_a, cluster_b, cluster_c]):
        buffer.evaluate_and_add(point)

    assignments_a = {buffer.assign_cluster(point)[0] for point in cluster_a}
    assignments_b = {buffer.assign_cluster(point)[0] for point in cluster_b}
    assignments_c = {buffer.assign_cluster(point)[0] for point in cluster_c}

    if len(assignments_a) != 1 or len(assignments_b) != 1 or len(assignments_c) != 1:
        return False, "points from the same synthetic cluster were split across different centroids"
    if len(assignments_a | assignments_b | assignments_c) != 3:
        return False, "two different synthetic clusters collapsed onto the same centroid"
    return True, "three well-separated clusters were assigned to three distinct centroids"


def test_sqir_threshold_resists_outliers():
    rng = np.random.default_rng(1)
    buffer = UnsupervisedSigLIPBuffer(n_clusters=1, embedding_dim=8, window_size=30, k=1.5)

    normal_scores = rng.normal(loc=0.5, scale=0.05, size=40).tolist()
    buffer.score_window = normal_scores[:30]
    threshold_before = buffer.admission_threshold()

    buffer.score_window.append(50.0)
    buffer.score_window.pop(0)
    threshold_with_outlier = buffer.admission_threshold()

    for score in normal_scores[30:]:
        buffer.score_window.append(score)
        buffer.score_window.pop(0)
    threshold_after_outlier_clears = buffer.admission_threshold()

    if threshold_with_outlier > threshold_before * 5:
        return False, f"a single outlier inflated the threshold too much: {threshold_before:.4f} -> {threshold_with_outlier:.4f}"
    if abs(threshold_after_outlier_clears - threshold_before) > 0.2:
        return False, "threshold did not recover once the outlier left the rolling window"
    return True, f"threshold stayed near {threshold_before:.4f} despite an outlier"


def test_cold_start_does_not_crash():
    buffer = UnsupervisedSigLIPBuffer(n_clusters=2, embedding_dim=8, window_size=20)
    rng = np.random.default_rng(2)

    try:
        for _ in range(3):
            buffer.evaluate_and_add(rng.normal(size=8))
    except Exception as exc:
        return False, f"raised an exception on cold start: {exc}"
    return True, "handled fewer than 4 samples without dividing by zero or crashing"


def test_balance_score_favors_underrepresented_cluster():
    buffer = UnsupervisedSigLIPBuffer(n_clusters=2, embedding_dim=8, window_size=20)
    buffer.cluster_counts = np.array([50.0, 0.0])

    score_overrepresented = buffer.balance_score(0)
    score_underrepresented = buffer.balance_score(1)

    if not (score_underrepresented > score_overrepresented):
        return False, f"expected underrepresented cluster to score higher, got {score_underrepresented} vs {score_overrepresented}"
    return True, f"underrepresented cluster scored {score_underrepresented:.3f} vs {score_overrepresented:.3f}"


def main():
    tests = [
        test_kmeans_assigns_synthetic_clusters_correctly,
        test_sqir_threshold_resists_outliers,
        test_cold_start_does_not_crash,
        test_balance_score_favors_underrepresented_cluster,
    ]

    results = [(test.__name__, *test()) for test in tests]

    print()
    for name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")

    failed_count = sum(1 for _, passed, _ in results if not passed)
    print(f"\n{len(results) - failed_count}/{len(results)} tests passed")
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
