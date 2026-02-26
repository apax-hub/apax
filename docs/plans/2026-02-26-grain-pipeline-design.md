# Design Document: Grain Data Pipeline for Apax

## Overview
This design introduces a new, JAX-native data pipeline using Google's **Grain** library. The primary goal is to provide a scalable, performant alternative to the existing TensorFlow-based pipeline while removing the project's dependency on TensorFlow.

## Architecture

### 1. Data Conversion: Structure of Arrays (SoA)
Instead of operating on a list of `ase.Atoms` objects, the pipeline will convert data into a dictionary of NumPy arrays (SoA) upon initialization.
- **Fixed-Padding SoA:** Arrays are pre-padded to the maximum number of atoms/neighbors in the dataset.
- **Ragged SoA:** Data is stored in its natural shape (e.g., as a dictionary of lists) to support variable-sized systems and bucketed padding.

### 2. Grain Components

#### `SoADataSource`
A custom implementation of `grain.python.RandomAccessDataSource`.
- **Inputs:** The SoA dictionary.
- **Functionality:** Efficiently retrieves a single sample by index.

#### `NeighborListTransform`
A `grain.python.MapTransform` that encapsulates the neighbor list calculation.
- **Logic:** Uses the existing `apax.data.preprocessing.compute_nl` function.
- **Benefit:** Leverages Grain's internal multiprocessing to calculate neighbor lists in parallel on the CPU during data loading.

#### `ApaxGrainDataLoader`
A high-level wrapper that configures the Grain pipeline.
- **Features:** Supports both fixed padding (Approach 1) and will be extended to support bucketed padding (Approach 2).
- **Shuffling:** Uses Grain's built-in shuffling mechanisms.
- **Sharding:** Native support for JAX sharding/multi-device training.

## Performance & Benchmarking
To ensure no regressions, a `benchmark_pipelines.py` script will be created in the project root.
- **Metrics:** Throughput (samples/sec) and latency (ms/batch).
- **Validation:** Compare batch contents between the legacy `CachedInMemoryDataset` and the new `ApaxGrainDataLoader` to ensure parity.

## Coexistence Strategy
The new pipeline will be implemented in `apax/data/grain_pipeline.py`. Existing modules will remain untouched. Users can opt-in to the new pipeline via configuration changes.

## Testing Strategy
- **Unit Tests:** Verify `SoADataSource` indexing and `NeighborListTransform` accuracy.
- **Integration Tests:** Run a minimal training loop with a small dataset (e.g., Ethanol) using the Grain pipeline.
- **Coverage:** Target >80% for all new components.
