from setuptools import setup
import os

# Get all Python modules in src/ directory
src_dir = "src"
py_modules = []
for file in os.listdir(src_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]  # Remove .py extension
        py_modules.append(module_name)

setup(
    name="scheduler",
    version="0.1.0",
    description="Generating Schedules for Robotic Workloads",
    py_modules=py_modules,
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "cvxpy",
        "matplotlib",
        "pandas",
    ],
)

