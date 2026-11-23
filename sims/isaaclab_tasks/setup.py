"""Installation script for the custom Isaac Lab tasks."""

from setuptools import find_packages, setup

setup(
    name="isaaclab_tasks",
    version="0.1.0",
    author="FreshScheduler Team",
    description="Custom Isaac Lab RL tasks for FreshScheduler project",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    zip_safe=False,
)
