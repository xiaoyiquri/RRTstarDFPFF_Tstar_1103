# System model 阶段说明

Phase 1 的 `linear_model.h` 是 double-integrator correctness baseline。后续系统 model 只组合 Flat State、DF、limits 和 DynoBench adapter，不复制 RRT*、PFF 或 SIMD 核心。
