# HAMi-core Memory Fix — Change Coverage

This document covers all code changes introduced in the `feat/dra-monitor-container-dirs` branch (commits `1cdc65f`..`747653b`) on top of `main` (commit `ec8979d`). These changes fix container-vs-host GPU memory tracking and add unconditional NVML memory monitoring.

---

## Summary

| Commit | Description |
|--------|-------------|
| `1cdc65f` | fix(mem): correct container vs host GPU memory and enforce memory limits |
| `896e22d` | fix(mem): restore monitor variable in NVML hook logging |
| `747653b` | feat(core): add `memory_monitor_watcher` thread for unconditional NVML memory tracking |

---

## Commit 1: `1cdc65f` — Fix container vs host GPU memory

### Problem

HAMi-core's memory tracking used `get_gpu_memory_usage()` which only counts `cudaMalloc`-intercepted allocations (internal tracking). This misses memory allocated by the CUDA runtime itself (context, module, buffers) and framework pre-allocations (e.g. PyTorch caching allocator). As a result:

- OOM checks were based on under-reported usage
- `cuMemGetInfo_v2` returned inaccurate free memory
- `nvmlDeviceGetMemoryInfo` could underflow when `limit < usage`
- NVML process matching compared container PIDs instead of host PIDs

### Changes

#### `src/allocator/allocator.c`

| Function | Change |
|----------|--------|
| `oom_check()` | Changed from `get_gpu_memory_usage(d)` to `get_gpu_memory_real_usage(d)` so OOM is triggered based on actual NVML-reported GPU memory, not just tracked `cudaMalloc` allocations. |

#### `src/cuda/memory.c`

| Function | Change |
|----------|--------|
| `cuMemGetInfo_v2()` | **Usage source**: Changed from `get_current_device_memory_usage(cuda_to_nvml_map(dev))` to `get_gpu_memory_real_usage(dev)` for accurate free memory. |
| | **Limit source**: Changed from `get_current_device_memory_limit(cuda_to_nvml_map(dev))` to `get_current_device_memory_limit(dev)` (removed redundant CUDA-to-NVML mapping). |
| | **Underflow guard**: `*free = (*total > usage) ? (*total - usage) : 0` instead of `*free = *total - usage` to prevent unsigned underflow. |
| | **Over-limit handling**: When `limit < usage`, instead of returning `CUDA_ERROR_INVALID_VALUE`, now returns `CUDA_SUCCESS` with `*free = 0` and `*total = limit`. This prevents applications from crashing on transient over-limit states. |

#### `src/multiprocess/multiprocess_memory_limit.c`

| Function | Change |
|----------|--------|
| `get_gpu_memory_real_usage()` | **New function.** Queries NVML directly via `nvml_get_device_memory_usage()` and also gets the tracked value from `get_gpu_memory_usage()`. Returns `max(nvml, tracked)` to be conservative — ensures neither NVML blind spots nor tracking gaps cause under-reporting. |
| `nvml_get_device_memory_usage()` | **Error handling**: Added `return 0` after `nvmlDeviceGetHandleByIndex` and `nvmlDeviceGetComputeRunningProcesses` failures (previously fell through with undefined behavior). |
| | **PID matching fix**: Changed `infos[i].pid != region->procs[slot].pid` to `infos[i].pid != region->procs[slot].hostpid`. NVML reports host PIDs, but `proc.pid` is the container-namespace PID from `getpid()`. The `hostpid` field holds the real host PID, which is what NVML returns. |
| | **Debug logging**: Added `LOG_DEBUG` for each matched process showing `hostpid` and `usedGpuMemory`. |
| `get_current_device_memory_usage()` | Changed from `get_gpu_memory_usage(dev)` to `get_gpu_memory_real_usage(dev)` so all callers get NVML-based usage. |

#### `src/multiprocess/multiprocess_memory_limit.h`

| Change |
|--------|
| Added declaration: `size_t get_gpu_memory_real_usage(const int dev);` |

#### `src/nvml/hook.c`

| Function | Change |
|----------|--------|
| `_nvmlDeviceGetMemoryInfo()` | **Removed unused variable**: Removed `monitor` variable and simplified logging from `"usage=%ld limit=%ld monitor=%ld"` to `"usage=%ld limit=%ld"`. |
| | **Underflow guard**: Both v1 (`nvmlMemory_t`) and v2 (`nvmlMemory_v2_t`) paths changed from `(limit - usage)` to `(limit > usage) ? (limit - usage) : 0` for the `free` field. |

---

## Commit 2: `896e22d` — Restore monitor variable in NVML hook

### Changes

#### `src/nvml/hook.c`

| Function | Change |
|----------|--------|
| `_nvmlDeviceGetMemoryInfo()` | Restored `monitor` variable and the 3-field log format `"usage=%ld limit=%ld monitor=%ld"` that was removed in the previous commit. This preserves the ability to debug monitoring values in logs. |

---

## Commit 3: `747653b` — Add `memory_monitor_watcher` thread

### Problem

The existing `utilization_watcher` thread only runs when SM utilization limits are configured (`get_current_device_sm_limit(0) > 0 && <= 100`). If no SM limit is set, the shared memory `monitorused[]` field is never populated with NVML data. The HAMi-DRA-monitor reads `monitorused[]` to expose `vGPU_device_memory_usage_real_in_MiB`, so without this thread, real GPU memory is always 0 in metrics.

### Changes

#### `src/multiprocess/multiprocess_utilization_watcher.c`

| Function | Change |
|----------|--------|
| `update_monitorused()` | **New static function.** Queries NVML for all GPU devices, calls `nvmlDeviceGetComputeRunningProcesses()` for each device, matches process host PIDs to registered processes via `find_proc_by_hostpid()`, and writes `infos[i].usedGpuMemory` into `proc->monitorused[cudadev]`. Runs under `lock_shrreg()`/`unlock_shrreg()` for thread safety. |
| `memory_monitor_watcher()` | **New static thread function.** Initializes NVML, then loops every 1 second: waits for host PID discovery (`pidfound`), then calls `update_monitorused()`. Runs unconditionally regardless of SM limit configuration. |
| `init_utilization_watcher()` | **Modified.** Now always spawns `memory_monitor_watcher` thread before the conditional `utilization_watcher` thread. This ensures `monitorused[]` is always populated for the external monitor to read. |

---

## Impact Matrix

| Component | Before | After |
|-----------|--------|-------|
| OOM detection | Based on `cudaMalloc` tracking only | Based on `max(NVML, tracked)` — catches framework allocations |
| `cuMemGetInfo_v2` free memory | Could underflow (wrap to huge number) | Clamped to 0 when over limit |
| `cuMemGetInfo_v2` over-limit | Returned `CUDA_ERROR_INVALID_VALUE` | Returns `CUDA_SUCCESS` with `free=0` |
| `nvmlDeviceGetMemoryInfo` free | Could underflow | Clamped to 0 |
| NVML PID matching | Used container PID (wrong in namespaced environments) | Uses host PID (`hostpid` field) |
| NVML error handling | Fell through on error | Returns 0 on error |
| `monitorused[]` population | Only when SM limits active | Always (via `memory_monitor_watcher`) |
| Monitor real memory metrics | 0 when no SM limit | Always available via NVML polling |

---

## Files Changed Summary

| File | Lines Added | Lines Removed |
|------|------------|--------------|
| `src/allocator/allocator.c` | 3 | 1 |
| `src/cuda/memory.c` | 9 | 4 |
| `src/multiprocess/multiprocess_memory_limit.c` | 24 | 3 |
| `src/multiprocess/multiprocess_memory_limit.h` | 1 | 0 |
| `src/multiprocess/multiprocess_utilization_watcher.c` | 63 | 0 |
| `src/nvml/hook.c` | 3 | 2 |
| **Total** | **103** | **10** |
