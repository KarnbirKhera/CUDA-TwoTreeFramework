# Two Tree Framework - CUDA
A systematic and pedagogical way to derive the correctness structure of a kernel before coding.

The first version of the Two Tree framework introduces a way to derive indexes in a beginner friendly way by treating the execution hierarchy and memory hierarchy within CUDA as trees to be mapped using parallelization and iterators. 

The framework introduces the following beginner friendly vocabulary which maps to the following CUDA concepts:
- Execution and Memory Tree -> Execution and Memory hierarchy
- AREA iterator -> A memory/execution iterator where we map our execution tree to our memory tree
- SLIDE iterator -> A memory/memory iterator where how many times does our divisor memory fit into our numerator ex. how many tiles does it take to map across our grid (grid/tile_size)

This iteration of the framework assumes the following:
- The kernel follows a simple structre of Load -> Compute -> Store
- The framework assumes the tensor is stored/can be accessed using simple arithmetic (Index * Stride + Offset)
