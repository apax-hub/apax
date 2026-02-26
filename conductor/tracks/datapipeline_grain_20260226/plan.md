# Implementation Plan: New Data Pipeline based on Grain

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a scalable, JAX-native data pipeline using Google Grain that replaces the TensorFlow dependency and supports both fixed and bucketed padding.

**Architecture:** Data is converted to a Structure of Arrays (SoA). A custom `SoADataSource` and `NeighborListTransform` are used within a Grain `DataLoader`.

**Tech Stack:** Python, Google Grain, JAX, NumPy, ASE.

---

## Phase 1: Core Grain Components [checkpoint: 61fa5f7]

### Task 1: Implement `SoADataSource` [x] 5267f22

**Files:**
- Create: `apax/data/grain_pipeline.py`
- Test: `tests/unit_tests/data/test_grain_pipeline.py`

**Step 1: Write the failing test**
Create a test that verifies `SoADataSource` can be initialized with a dictionary of arrays and retrieves the correct item by index.

**Step 2: Run test to verify it fails**
`uv run pytest tests/unit_tests/data/test_grain_pipeline.py::test_soa_datasource -v`

**Step 3: Write minimal implementation**
Implement `SoADataSource` inheriting from `grain.python.RandomAccessDataSource`.

**Step 4: Run test to verify it passes**
`uv run pytest tests/unit_tests/data/test_grain_pipeline.py::test_soa_datasource -v`

**Step 5: Commit**
`git add apax/data/grain_pipeline.py tests/unit_tests/data/test_grain_pipeline.py`
`git commit -m "feat(data): Implement SoADataSource for Grain pipeline"`

---

### Task 2: Implement `NeighborListTransform` [x] fd1c794

**Files:**
- Modify: `apax/data/grain_pipeline.py`
- Test: `tests/unit_tests/data/test_grain_pipeline.py`

**Step 1: Write the failing test**
Create a test that applies `NeighborListTransform` to a sample and verifies the `idx` and `offsets` keys are present and correct.

**Step 2: Run test to verify it fails**
`uv run pytest tests/unit_tests/data/test_grain_pipeline.py::test_nl_transform -v`

**Step 3: Write minimal implementation**
Implement `NeighborListTransform` inheriting from `grain.python.MapTransform`.

**Step 4: Run test to verify it passes**
`uv run pytest tests/unit_tests/data/test_grain_pipeline.py::test_nl_transform -v`

**Step 5: Commit**
`git add apax/data/grain_pipeline.py tests/unit_tests/data/test_grain_pipeline.py`
`git commit -m "feat(data): Implement NeighborListTransform for Grain pipeline"`

---

## Phase 2: Pipeline ## Phase 2: Pipeline & Benchmarking Benchmarking [checkpoint: 34418c6]

### Task 3: Implement `ApaxGrainDataLoader` (Fixed Padding) [x] 24ea328

**Files:**
- Modify: `apax/data/grain_pipeline.py`
- Test: `tests/unit_tests/data/test_grain_pipeline.py`

**Step 1: Write the failing test**
Create a test that initializes `ApaxGrainDataLoader` and iterates through a few batches, checking shapes.

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**
Implement `ApaxGrainDataLoader` using `grain.python.DataLoader`.

**Step 4: Run test to verify it passes**

**Step 5: Commit**
`git add apax/data/grain_pipeline.py tests/unit_tests/data/test_grain_pipeline.py`
`git commit -m "feat(data): Implement ApaxGrainDataLoader with fixed padding"`

---

### Task 4: Implement Benchmark Script [x] 9b82898

**Files:**
- Create: `benchmark_pipelines.py`

**Step 1: Write the script**
Implement a script that compares `CachedInMemoryDataset` and `ApaxGrainDataLoader` on a dummy dataset (or Ethanol if available).

**Step 2: Run benchmark**
`uv run python benchmark_pipelines.py`

**Step 3: Commit**
`git add benchmark_pipelines.py`
`git commit -m "test(data): Add data pipeline benchmark script"`

---

## Phase 3: Advanced Features & Integration

### Task 5: Implement Bucketed Padding (Approach 2)
(Details to be expanded after benchmarking Phase 2)

### Task 6: Integration with Training Loop
(Details to be expanded)
