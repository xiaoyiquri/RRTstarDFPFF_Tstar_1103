# DynoBench adapter 阶段占位

这里将封装 Unicycle、Quad2D、Quad3D 的 DynoBench API。Core、PFF 和 SIMD 不应 include DynoBench 内部模型头文件；adapter 需要独立检查 finite、state/control bounds、真实 step 残差和碰撞。
