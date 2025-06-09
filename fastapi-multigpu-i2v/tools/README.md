# Tools 工具使用指南

## 🛠️ 工具概览

### 1. `verify_structure.py` - 项目结构验证
完整验证项目文件结构、依赖关系和配置正确性。

```bash
# 完整验证
python3 tools/verify_structure.py

# 功能：
# - 检查必需文件和目录
# - 验证Python导入
# - 检查依赖包
# - 验证配置文件
# - 检查运行时环境
# - 验证文件权限
```

### 2. `config_generator.py` - 配置生成器
根据硬件环境自动生成最优配置。

```bash
# 生成生产配置
python3 tools/config_generator.py --template production

# 生成开发配置并导出环境文件
python3 tools/config_generator.py --template development --export-env

# 自定义配置
python3 tools/config_generator.py --custom '{"max_concurrent_tasks": 2, "t5_cpu": false}'

# 指定模型路径和端口
python3 tools/config_generator.py --model-path /path/to/model --port 8080
```

### 3. `benchmark.py` - 性能基准测试
测试API服务的性能指标。

```bash
# 响应时间测试
python3 tools/benchmark.py --response-test 10

# 负载测试
python3 tools/benchmark.py --load-test --concurrent-users 5

# 完整测试并导出结果
python3 tools/benchmark.py --response-test 5 --load-test --export results.json
```

### 4. `health_monitor.py` - 健康监控
监控服务健康状态和性能指标。

```bash
# 单次健康检查
python3 tools/health_monitor.py --mode check

# 持续监控 30 分钟
python3 tools/health_monitor.py --mode monitor --duration 1800

# 导出监控数据
python3 tools/health_monitor.py --mode monitor --duration 600 --export health_data.json
```

## 🚀 使用场景

### 项目部署前检查
```bash
# 1. 验证项目结构
python3 tools/verify_structure.py

# 2. 生成配置
python3 tools/config_generator.py --template production --export-env

# 3. 启动服务
source config_production.env
./scripts/start_service_general.sh
```

### 性能测试和监控
```bash
# 启动监控（后台）
python3 tools/health_monitor.py --mode monitor --duration 3600 &

# 运行性能测试
python3 tools/benchmark.py --load-test --concurrent-users 10

# 查看监控结果
fg  # 回到监控进程
```

### 问题排查
```bash
# 检查项目完整性
python3 tools/verify_structure.py

# 健康检查
python3 tools/health_monitor.py --mode check

# 配置诊断
python3 tools/config_generator.py --template development
```

## 📊 输出示例

### 项目验证成功
```
🚀 FastAPI Multi-GPU I2V Project Structure Verification
================================================================================
📄 Checking required files...
✅ Found 15 required files

🗂️  Checking directory structure...
✅ All required directories exist

🐍 Checking Python imports...
✅ All imports successful

📋 Verification Summary
================================================================================
Files                ✅ PASS
Directories          ✅ PASS
Imports              ✅ PASS
Dependencies         ✅ PASS
Configuration        ✅ PASS
Environment          ✅ PASS
Permissions          ✅ PASS
--------------------------------------------------------------------------------
Total Checks: 42
Categories: 7/7 passed
🎉 PROJECT VERIFICATION PASSED!
✅ Your project structure is ready for deployment
```

### 配置生成示例
```
🔍 Detected: npu x 8 (32.0GB)

📋 Generated Configuration
============================================================
Model Configuration:
  Model Path:        /data/models/modelscope/hub/Wan-AI/Wan2.1-I2V-14B-720P
  T5 CPU Mode:       true
  DiT FSDP:          true
  VAE Parallel:      true
  Max Concurrent:    4

Environment Variables:
  T5_CPU=true
  DIT_FSDP=true
  VAE_PARALLEL=true
  ASCEND_LAUNCH_BLOCKING=0
  HCCL_TIMEOUT=2400
```

## ⚠️ 常见问题

### 1. 导入错误
```
❌ Import failed: Project schemas - No module named 'schemas'
```
**解决方案**：确保在项目根目录执行，检查 PYTHONPATH

### 2. 权限问题
```
⚠️  Script start_service_general.sh is not executable
```
**解决方案**：工具会自动尝试修复，或手动执行 `chmod +x scripts/*.sh`

### 3. 依赖缺失
```
❌ Critical dependency missing: torch
```
**解决方案**：检查 requirements.txt 和虚拟环境

## 🔧 扩展和定制

### 添加新的验证检查
在 `verify_structure.py` 中的 `run_all_checks()` 方法添加新检查：

```python
results["new_check"] = self.check_new_feature()
```

### 自定义配置模板
在 `config_generator.py` 中添加新模板：

```python
self.templates["custom"] = {
    "t5_cpu": False,
    "max_concurrent_tasks": 6,
    # ... 其他配置
}
```

### 添加新的监控指标
在 `health_monitor.py` 中扩展 `HealthMetrics` 类。

## 🎯 最佳实践

1. **部署前必检**：始终运行 `verify_structure.py`
2. **配置优化**：根据硬件使用 `config_generator.py`
3. **持续监控**：生产环境使用 `health_monitor.py`
4. **定期基准**：使用 `benchmark.py` 检测性能退化