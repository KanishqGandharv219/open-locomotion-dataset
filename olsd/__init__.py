"""
OLSD — Open Locomotion Skills Dataset

A unified benchmark for legged robot locomotion across diverse morphologies
and terrains.

Quick start:
    import olsd

    # Load a dataset
    dataset = olsd.load("./data/olsd-v0.1")

    # Filter
    quads = dataset.filter(morphology="quadruped")

    # Iterate
    for episode in quads:
        print(episode.metadata.robot.robot_id, episode.n_steps)
"""

__version__ = "0.1.0"

from olsd.sdk.loader import OLSDDataset, load

__all__ = ["load", "OLSDDataset", "__version__"]
