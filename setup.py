from setuptools import setup
import os

# Get all top-level Python modules in xpu-rt/ directory.
# This repo uses module-style imports (e.g. "from workload import ..."),
# so we expose each .py file in xpu-rt as a top-level module.
src_dir = "xpu-rt"
py_modules = []
for file in os.listdir(src_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]  # Remove .py extension
        py_modules.append(module_name)

setup(
    name="xpu-rt",
    version="0.1.0",
    description="An adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.",
    py_modules=py_modules,
    package_dir={"": "xpu-rt"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "cvxpy",
        "matplotlib",
        "pandas",
    ],
)
