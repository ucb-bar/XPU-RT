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
    # Optional solver backends. Neither is a hard dependency: the registry
    # imports them lazily and `scripts/profile_schedulers.py` probes for them
    # and reports "unavailable" rather than failing. They are named here so
    # that "the sweep only ran four of six policies" has a one-line fix
    # (`pip install -e '.[solvers]'`) instead of a guess.
    #
    # MOSEK additionally needs a licence file (~/mosek/mosek.lic or
    # $MOSEKLM_LICENSE_FILE); installing the wheel alone is not enough.
    extras_require={
        "solvers": ["ortools", "mosek"],
    },
)
