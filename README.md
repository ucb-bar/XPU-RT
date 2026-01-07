# RobotSchedule
Generating Schedules for Robotic Workloads

## Installation

Install the package in editable mode:
```bash
pip install -e .
```

## Examples

The repository includes several example scripts in `src/scripts/`:

### Basic Scheduling Examples

**`testing.py`** - Parameter sweep for scheduling workloads with transfer times
- Performs a parameter sweep over different numbers of jobs and machines
- Generates synthetic sequential workloads with transfer times
- Saves runtime results to CSV and schedule plots to `plots/` directory
- Run: `python src/scripts/testing.py`

**`testing_simple.py`** - Simple scheduling without transfer times
- Demonstrates scheduling with zero transfer times (simplified problem)
- Creates a workload with 4 sequential jobs on 3 machines (CPU, GPU, FPGA)
- Saves plot to `plots/simple_test.png`
- Run: `python src/scripts/testing_simple.py`

### Advanced Scheduling Examples

**`testing_iree.py`** - Schedule IREE dispatch graphs on dual-core device
- Schedules real neural network dispatch graphs (Fast, Dronet, and 5 MLP instances)
- Models a dual-core device with CPU_P (performant, 1.5x faster) and CPU_E (efficient)
- Creates dependency chains: Fast → Dronet, and MLP0 → MLP1 → MLP2 → MLP3 → MLP4
- All MLP instances share the same color in the visualization
- Saves plot to `plots/iree_combined_schedule.png`
- Run: `python src/scripts/testing_iree.py`

**`packing_demo.py`** - Demonstrates greedy vs. convex packing algorithms
- Compares greedy and convex packing approaches for workload scheduling
- Shows trade-offs between solution quality and computation time
- Saves plots to `plots/greedy_schedule.pdf` and `plots/convex_schedule.pdf`
- Run: `python src/scripts/packing_demo.py`

**`additional_obj_demo.py`** - Scheduling with additional objectives
- Demonstrates scheduling with nominal start times and periodic constraints
- Useful for real-time scheduling scenarios
- Saves plot to `plots/additional_objectives_schedule.png`
- Run: `python src/scripts/additional_obj_demo.py`

## Output

Most examples generate:
- Schedule visualizations saved to `plots/` directory
- Runtime statistics printed to console
- Some examples also save CSV files with performance metrics
