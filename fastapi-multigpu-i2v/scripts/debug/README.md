# Debug Tools 调试工具使用指南

## 🛠️ 工具概览

### 1. `debug_device.py` - 设备检测调试
```bash
# 完整设备检测测试
python3 scripts/debug/debug_device.py

# 功能：
# - 基础设备检测
# - 设备详细信息
# - 内存操作测试
# - 管道兼容性检查
# - 环境变量检查
```

### 2. `debug_pipeline.py` - 管道调试
```bash
# 综合管道测试
python3 scripts/debug/debug_pipeline.py

# 快速测试
python3 scripts/debug/debug_pipeline.py --mode quick

# 自定义模型路径
python3 scripts/debug/debug_pipeline.py --model-path /path/to/model

# 功能：
# - 管道创建测试
# - 基本方法测试
# - 请求对象创建
# - 清理功能测试
```

### 3. `debug_memory.py` - 内存监控
```bash
# 查看当前内存状态
python3 scripts/debug/debug_memory.py --mode status

# 连续监控 30 秒
python3 scripts/debug/debug_memory.py --mode monitor --duration 30

# 模型加载内存测试
python3 scripts/debug/debug_memory.py --mode model-test

# 内存压力测试
python3 scripts/debug/debug_memory.py --mode stress-test

# 导出数据到 CSV
python3 scripts/debug/debug_memory.py --mode monitor --duration 60 --export memory_data.csv
```

### 4. `debug_t5_warmup.py` - T5预热调试
```bash
# T5 CPU 预热测试
python3 scripts/debug/debug_t5_warmup.py

# 自定义预热步数
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 5

# 跳过资源检查
python3 scripts/debug/debug_t5_warmup.py --skip-resource-check
```

## 🚀 使用场景

### 服务启动前检查
```bash
# 1. 检查设备
python3 scripts/debug/debug_device.py

# 2. 检查管道
python3 scripts/debug/debug_pipeline.py --mode quick

# 3. 检查内存
python3 scripts/debug/debug_memory.py --mode status
```

### 问题排查
```bash
# NPU 问题排查
NPU_VISIBLE_DEVICES=0 python3 scripts/debug/debug_device.py

# 内存不足问题
python3 scripts/debug/debug_memory.py --mode stress-test

# T5 预热问题
python3 scripts/debug/debug_t5_warmup.py --warmup-steps 1
```

### 性能监控
```bash
# 长期内存监控
python3 scripts/debug/debug_memory.py --mode monitor --duration 3600 --interval 10 --export long_monitor.csv

# 模型加载分析
python3 scripts/debug/debug_memory.py --mode model-test
```

## 📊 输出示例

### 设备检测成功示例
```
🔍 Device Detection Debug Tool - Complete Version
============================================================
🔍 Basic Device Detection
----------------------------------------
✅ Detected device: npu
✅ Device count: 8
✅ Backend: torch_npu

📊 Device Details
----------------------------------------
Device 0:
  Name: Ascend910A
  Memory: 0.05GB allocated, 0.10GB reserved
  NPU device available: True
...

📋 Test Summary
============================================================
Basic Detection          ✅ PASS
Device Details           ✅ PASS
Memory Operations        ✅ PASS
Pipeline Compatibility   ✅ PASS
----------------------------------------
Total: 4/4 tests passed
🎉 All device detection tests PASSED!
```

## ⚠️ 常见问题

### 1. 导入错误
```
❌ Import failed: No module named 'utils.device_detector'
```
**解决方案**：检查项目路径和依赖安装

### 2. 设备不可用
```
❌ No CUDA devices detected!
```
**解决方案**：检查驱动安装和设备可见性

### 3. 内存不足
```
❌ CUDA out of memory
```
**解决方案**：降低并发数或使用内存监控工具分析

## 🔧 自定义扩展

可以通过修改调试工具来添加新的测试功能，所有工具都有清晰的模块化结构。