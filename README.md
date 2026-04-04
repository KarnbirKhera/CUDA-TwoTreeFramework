# Two Tree Framework — CUDA

A systematic and pedagogical approach to deriving the correct index structure of a CUDA kernel before writing a single line of code.

---

## Motivation

This framework was born out of a genuine struggle. When first approaching tiled GEMM in CUDA, the core problem wasn't the math or the parallelism. It was the indexes.

Writing a kernel requires answering questions like:
- Where exactly in global memory does this thread read from?
- How does a 2D block of threads collectively fill a 2D shared memory tile?
- How does each thread know which element of the output matrix it is responsible for?

The usual approach is to pattern-match against existing kernels or memorize formulas. But this breaks down quickly when parameters change, when something is subtly wrong, or when you need to reason about a new access pattern from scratch.

The turning point came from a simple observation that is easy to miss:

> *Everything is technically flattened.*

Every memory access in CUDA, no matter how 2D or 3D it feels conceptually, ultimately reduces to a single flat address computed as `Coordinate * Stride + Offset`. The execution hierarchy and the memory hierarchy are both just trees of nested sizes, and mapping one onto the other is a mechanical process once you see it clearly.

The Two Tree Framework is the formalization of that realization. Rather than a set of rules to memorize, it is a procedure you can apply from scratch, building the table, deriving the iterators, and reading off the indexes, so that every term in every expression has a clear and traceable reason for being there.

---

## What is it?

The Two Tree Framework is a mental model and derivation system for reasoning about CUDA indexes in tiled GEMM (and similar) kernels. Instead of guessing at index expressions or memorizing patterns, the framework gives you a structured procedure to *derive* every index from first principles, verifying correctness before a single line of kernel code is written.

The core insight is that CUDA has two parallel hierarchies that must be mapped onto each other:
- An **Execution Tree**: Grid -> Block -> Thread
- A **Memory Tree**: Global -> Shared -> Register

By filling in sizes, iterators, and indexes into a unified Two Tree Table, you can read off the formula for any index directly, with no guesswork and no memorization.

---

## The Five Sections

### Section One - Building the Two Trees

The first step is constructing the Two Tree Table, which sits at the heart of the framework. The table has two sides:

**Execution Tree** columns: Level | Size | Index (Column, Row)
- Grid: size = `gridDim`, indexes = `blockIdx.x`, `blockIdx.y`
- Block: size = `blockDim`, indexes = `threadIdx.x`, `threadIdx.y`
- Thread: size = `1`, indexes = `regCol`, `regRow`

**Memory Tree** columns: Index (Column, Row) | Size | Level
- Global: size = M, N, K, the full problem dimensions
- Shared: size = TILE_SIZE, the shared memory tile
- Register: size = WORK_PER_THREAD, the per-thread workload

With both sides filled in, the next task is to **bridge** the execution tree to the memory tree so that every element is processed. This is done using **iterators** (for loops). To determine what kind of iterator connects each execution level to its corresponding memory level, the framework asks a key diagnostic question:

> *Does this execution level, on each iteration, work through its existing memory region, or does it require its memory level to provide a new region entirely?*

The answer determines which of two iterator formulas to use:
- **Moving data (SLIDE)**: `Current Memory Size / TILE_SIZE`, where the execution level slides across memory one tile at a time
- **Processing data (AREA)**: `Current Memory Size / Current Execution Size`, where the execution level sweeps through a region it already holds

Applying this question to each level in a tiled GEMM with sizes Grid=32x32, Block=8x8, Thread=1, Global=1024x1024, Shared=32x32, Register=4x4:

| Execution Level | Iterator Type | Formula | Result | Iterator Name |
|---|---|---|---|---|
| Grid -> Global | SLIDE (data movement, slides across K) | 1024 / 32 | 32 iterations | `tileId` |
| Block -> Shared | AREA (data processing, covers 2D tile) | (32x32) / (8x8) | 16 iterations | `stride` |
| Thread -> Register | AREA (data processing, covers 2D register tile) | (4x4) / (1x1) | 16 iterations | `regCol`, `regRow` |

For the thread-level iterator, since it covers a 2D register tile, two separate iterators are needed, one per dimension. Since these iterators directly step through every element in the 4x4 register tile, they double as the register-level memory indexes (`regCol` and `regRow`).

---

### Section Two - The Three Phases

Every index belongs to exactly one of three kernel phases. Knowing which phase an index belongs to determines how and where to derive it:

**Load Phase** - Data moves from global memory to shared memory to registers. This is where most index derivation happens, since we need to correctly address source data at every level of the memory hierarchy.

**Compute Phase** - Pure arithmetic entirely within shared memory and registers. All indexes needed here were already derived during the Load phase, plus each thread derives its own position within shared memory directly.

**Store Phase** - Results are written from registers back to global memory. This requires deriving the output matrix indexes (`cCol`, `cRow`).

*Note: It is generally good practice to place a `__syncthreads()` barrier between phases.*

---

### Section Three - Derivation Tools

Before deriving any index, you must first ask:

> *Is this index targeting global/register memory, or shared memory in our memory tree?*

#### For Global or Register Memory - The Four Questions

Indexes at the global or register level are built level-by-level using the recurring formula:

```
Index = execution_index x stride + next_level_index
```

Four questions scaffold this formula:

1. **Which memory level does our index terminate at?**
   Tells you how far the index needs to reach. Does it stop at shared memory, or go all the way to global?

2. **Which execution level is at the same level as our target memory level, and what is its index?**
   Tells you which execution index (`blockIdx`, `threadIdx`, etc.) appears in the formula.

3. **What is the memory size of the level *below* the current memory level?**
   This is the stride, the size of the sub-region the index must skip over per step.

4. **What is the memory index at our target memory level?**
   This is the final term in the formula, the offset within the current level, which may itself require repeating questions 1-3 recursively until the bottom of the tree is reached.

*Note: These four questions are scaffolding to build intuition. With practice, you can read the index formula directly from the Two Tree Table without explicitly asking them, as demonstrated in Section Five.*

#### For Shared Memory - Flatten -> Stride -> Unflatten

Shared memory requires a different approach because a 2D block of threads must collectively load a 2D tile, which cannot be done with simple 1D arithmetic. The pattern is:

**Flatten**: Collapse the 2D thread position into a single 1D local ID so arithmetic is possible.
```
localId = threadIdx.y x blockDim.x + threadIdx.x
```
For an 8x8 block: range 0-63. Max check: 7x8+7 = 63 ✓

**Stride**: Step through all elements of the shared memory tile across multiple iterations using the stride iterator, since 64 threads cannot cover 1024 elements in one pass.
```
flatIdx = stride x (blockDim.x x blockDim.y) + localId
```
For 16 strides: range 0-1023. Max check: 15x64+63 = 1023 ✓

**Unflatten**: Convert the 1D flat index back to 2D shared memory coordinates, because shared memory is addressed in 2D.
```
sCol = flatIdx % TILE_SIZE
sRow = flatIdx / TILE_SIZE
```
For TILE_SIZE=32: range 0-31. Max check: 1023%32 = 31, 1023/32 = 31 ✓

Every parameter this pattern needs is already present in the Two Tree Table, so no new quantities need to be derived.

---

### Section Four - Deriving the Indexes

With the Two Tree Table complete and the derivation tools established, indexes are derived in order from the bottom of the memory tree upward, following the Load -> Compute -> Store phase structure.

#### Load Phase

**regCol and regRow** (Register level)
Since `regCol` and `regRow` are the thread-level iterators themselves, they require no further derivation and are read directly from the table.

**sCol and sRow** (Shared memory level - Flatten -> Stride -> Unflatten)
These target shared memory, so the Flatten -> Stride -> Unflatten pattern applies. The 64 threads of the 8x8 block collectively load all 1024 elements of the 32x32 shared tile across 16 stride iterations, producing `sCol` and `sRow` as derived above.

**aCol and aRow** (Global level - Four Questions applied to Matrix A)
Both target global memory, so the four questions apply:
- Q1: Global level
- Q2: Grid maps to global; execution indexes are `blockIdx.x` and `blockIdx.y`
- Q3: Memory level below global is shared; stride = `TILE_SIZE`
- Q4: Target is shared level; next-level indexes are `sCol` and `sRow`

Matrix A slides across the K dimension (columns) while staying fixed on M (rows):
```
aCol = tileId x TILE_SIZE + sCol
aRow = blockIdx.y x TILE_SIZE + sRow
```
Max check: 31x32+31 = 1023 ✓, properly maps to Global (1024x1024)

**bCol and bRow** (Global level - Four Questions applied to Matrix B)
Same process as A, but Matrix B slides across K (rows) while staying fixed on N (columns):
```
bCol = blockIdx.x x TILE_SIZE + sCol
bRow = tileId x TILE_SIZE + sRow
```
Max check: 31x32+31 = 1023 ✓, properly maps to Global (1024x1024)

#### Compute Phase

**sharedCol and sharedRow** (Shared memory level - Four Questions)
Each thread derives its own position within the shared memory tile directly:
- Q1: Shared level
- Q2: Block maps to shared; execution indexes are `threadIdx.x` and `threadIdx.y`
- Q3: Memory level below shared is register; stride = `WORK_PER_THREAD`
- Q4: Target is register level; next-level indexes are `regCol` and `regRow`

```
sharedCol = threadIdx.x x WORK_PER_THREAD + regCol
sharedRow = threadIdx.y x WORK_PER_THREAD + regRow
```
Max check: 7x4+3 = 31 ✓, properly maps to TILE_SIZE (32)

Access patterns:
- `A_shared[sharedRow][k]` selects 4 rows at a fixed column K
- `B_shared[k][sharedCol]` selects 4 columns at a fixed row K

#### Store Phase

**cCol and cRow** (Global level - read directly from the table)
By this point the intuition from the four questions is fully internalized, so these can be read directly from the Two Tree Table by tracing each column down:

```
cCol = blockIdx.x x TILE_SIZE + threadIdx.x x WORK_PER_THREAD + regCol
cRow = blockIdx.y x TILE_SIZE + threadIdx.y x WORK_PER_THREAD + regRow
```
Max check: 31x32 + 7x4 + 3 = 1023 ✓, properly maps to Global (1024x1024)

---

### Section Five - Reading the Table Directly

The four questions are scaffolding, not a permanent crutch. Once you have applied them a few times, the Two Tree Table becomes self-sufficient. You can trace any column index from top to bottom, multiplying each execution index by the memory stride of the level below, and summing the terms. The derivation of `cCol` and `cRow` above demonstrates this: no questions were needed, just a direct read of the table.

---

## Complete Index Summary

```cuda
// Load Phase
int localId   = threadIdx.y * blockDim.x + threadIdx.x;
int flatIdx   = stride * (blockDim.x * blockDim.y) + localId;
int sCol      = flatIdx % TILE_SIZE;
int sRow      = flatIdx / TILE_SIZE;

int aCol      = tileId * TILE_SIZE + sCol;
int aRow      = blockIdx.y * TILE_SIZE + sRow;

int bCol      = blockIdx.x * TILE_SIZE + sCol;
int bRow      = tileId * TILE_SIZE + sRow;

// Compute Phase
int sharedCol = threadIdx.x * WORK_PER_THREAD + regCol;
int sharedRow = threadIdx.y * WORK_PER_THREAD + regRow;

// Store Phase
int cCol      = blockIdx.x * TILE_SIZE + threadIdx.x * WORK_PER_THREAD + regCol;
int cRow      = blockIdx.y * TILE_SIZE + threadIdx.y * WORK_PER_THREAD + regRow;
```

---

## Framework Vocabulary Reference

| Term | Meaning |
|---|---|
| Execution Tree | The CUDA execution hierarchy: Grid -> Block -> Thread |
| Memory Tree | The CUDA memory hierarchy: Global -> Shared -> Register |
| SLIDE iterator | Data-movement iterator where the execution level requires new data each iteration. Bound = Memory Size / TILE_SIZE |
| AREA iterator | Data-processing iterator where the execution level works through data it already has. Bound = Memory Size / Execution Size |
| Flatten | Collapse a 2D thread position into a 1D local ID for arithmetic |
| Stride | Step index used to reuse threads across multiple passes of a memory region |
| Unflatten | Convert a 1D flat index back into 2D memory coordinates |
| Load Phase | Data movement from global to shared to registers; most index derivation happens here |
| Compute Phase | Pure arithmetic in shared memory and registers; indexes already derived |
| Store Phase | Writing results back to global memory; requires output matrix indexes |

---

## Who is this for?

Anyone learning CUDA who has stared at an index expression like:
```cuda
cRow = blockIdx.y * TILE_SIZE + threadIdx.y * WORK_PER_THREAD + regRow;
```
...and wondered where each term actually comes from. The Two Tree Framework makes that derivation feel inevitable rather than mysterious, and gives you the tools to produce it yourself from scratch for any kernel that fits the model.

For more context on what motivated this framework, see `MLSys2026-9Week-LearningPlan`, `week2-gemm`, and `PARKING_LOT.md` on the respective `MLSys2026-9Week-LearningPlan` repo.

---

## Current Assumptions

This version of the framework assumes:
- The kernel follows a Load -> Compute -> Store structure
- Tensors are stored and accessed using simple row-major arithmetic: `Index x Stride + Offset`
- The execution model is Grid / Block / Thread, as warp-level operations such as tensor cores introduce additional iterator types that require extending the framework beyond this model


---

## Scope and Intent

This framework targets GPU features up to sm_70 (Volta) and focuses exclusively on **correctness**: deriving the right index expressions and memory staging for tiled kernels. It does not address performance tuning (occupancy, bank conflicts, instruction throughput), which requires profiling.

Every version of this framework, current and future, is refined with a single pedagogical goal: to lower the barrier to entry for CUDA, not by merely presenting the correct answer, but by making the path to that answer feel learnable.

It is worth being explicit about what this framework is **not**:
- It is **not** a replacement for production frameworks like CUTLASS, cuTe, or Triton.
- It is **not** a substitute for profiling, which tells you how and why your kernel performs on specific hardware.

What it **is** meant to be is a **teaching tool**. Understanding *why* a kernel is correct and understanding *how* it performs are two separate problems; this framework addresses the former. Given a tensor layout and a set of operations, it provides a systematic way to derive kernel structure from first principles. It works in both directions:

- **Top-down**: From geometry and problem structure to kernel code.
- **Bottom-up**: From existing kernel code back to the geometry, so beginners can understand why the structure is the way it is.

The goal is not to abstract away CUDA, but to make its underlying logic visible and learnable.
