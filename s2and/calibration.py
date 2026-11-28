"""Training-only setup for clustering calibration."""

from typing import Any


def default_cluster_search_space() -> dict[str, Any]:
    """Build the default FastCluster calibration space when training starts.

    Returns:
        A Hyperopt space with EPS sampled uniformly between zero and one.
    """
    from hyperopt import hp

    return {"eps": hp.uniform("eps", 0, 1)}
