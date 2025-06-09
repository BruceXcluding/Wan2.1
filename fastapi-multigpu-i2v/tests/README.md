# Tests Directory

这个目录包含了项目的所有测试脚本，整合了原 `scripts/debug/` 目录的功能。

## 🧪 测试脚本说明

### 🔍 `test_env.py` - 环境测试工具
全面的环境检查工具，检查项目结构、模块导入、硬件环境等。

```bash
# 完整测试
python3 tests/test_env.py

# 快速测试
python3 tests/test_env.py --quick
```

### 🖥️ `test_device_detailed.py` - 设备详细测试
详细的设备检测和性能测试工具。

```bash
# 运行详细设备测试
python3 tests/test_device_detailed.py
```

### 🧠 `test_memory.py` - 内存监控测试
内存使用监控和分析工具。

```bash
# 查看当前内存状态
python3 tests/test_memory.py --mode status

# 连续监控 30 秒
python3 tests/test_memory.py --mode monitor --duration 30

# 模型加载内存测试
python3 tests/test_memory.py --mode model-test

# 内存压力测试
python3 tests/test_memory.py --mode stress-test

# 导出数据到 CSV
python3 tests/test_memory.py --mode monitor --duration 60 --export memory_data.csv
```

### 🔧 `test_pipeline.py` - 管道测试
管道创建和功能测试工具。

```bash
# 综合管道测试
python3 tests/test_pipeline.py

# 快速测试
python3 tests/test_pipeline.py --mode quick

# 自定义模型路径
python3 tests/test_pipeline.py --model-path /path/to/model
```

### 🔥 `test_warmup.py` - T5 预热测试
T5 CPU 预热功能测试。

```bash
# T5 CPU 预热测试
python3 tests/test_warmup.py

# 自定义预热步数
python3 tests/test_warmup.py --warmup-steps 5

# 跳过资源检查
python3 tests/test_warmup.py --skip-resource-check
```

## 🚀 推荐的测试流程

### 1. 首次部署时
```bash
# 步骤1: 环境检查
python3 tests/test_env.py

# 步骤2: 设备检测
python3 tests/test_device_detailed.py

# 步骤3: 内存状态
python3 tests/test_memory.py --mode status

# 步骤4: 管道测试
python3 tests/test_pipeline.py --mode quick

# 步骤5: T5 预热测试
python3 tests/test_warmup.py --warmup-steps 2
```

### 2. 日常检查
```bash
# 快速检查
python3 tests/test_env.py --quick

# 或者一键检查脚本
./tests/quick_check.sh  # 如果创建的话
```

### 3. 问题排查
```bash
# 设备问题
python3 tests/test_device_detailed.py

# 内存问题
python3 tests/test_memory.py --mode stress-test

# 管道问题
python3 tests/test_pipeline.py

# T5 预热问题
python3 tests/test_warmup.py --warmup-steps 1
```

### 4. 性能监控
```bash
# 长期内存监控
python3 tests/test_memory.py --mode monitor --duration 3600 --interval 10 --export performance.csv

# 模型加载分析
python3 tests/test_memory.py --mode model-test
```

### 5. 服务启动前
```bash
# 完整检查
python3 tests/test_env.py --quick && \
python3 tests/test_memory.py --mode status && \
./scripts/start_service_general.sh
```

## 📊 测试结果说明

- ✅ 测试通过
- ❌ 测试失败  
- ⚠️ 警告（不影响基本功能）
- ⚪ 信息（可选功能）

## 🔧 故障排除

如果测试失败，请按以下顺序检查：

1. **Python 环境和依赖**
   ```bash
   python3 --version
   pip install -r requirements.txt
   ```

2. **项目结构完整性**
   ```bash
   python3 tests/test_env.py
   ```

3. **硬件驱动程序**
   ```bash
   python3 tests/test_device_detailed.py
   ```

4. **内存资源**
   ```bash
   python3 tests/test_memory.py --mode status
   ```

5. **环境变量配置**
   ```bash
   python3 tests/test_env.py --quick
   ```

## 📁 文件说明

- `test_env.py` - 环境和导入测试
- `test_device_detailed.py` - 设备详细测试
- `test_memory.py` - 内存监控测试
- `test_pipeline.py` - 管道功能测试
- `test_warmup.py` - T5 预热测试
- `README.md` - 本说明文档

## 🎯 测试覆盖范围

✅ **环境检查**
- Python 环境和版本
- 依赖包安装状态
- 项目结构完整性
- 环境变量配置

✅ **硬件检测**
- NPU/CUDA/CPU 设备检测
- 设备数量和属性
- 内存容量和使用情况
- 设备驱动状态

✅ **功能测试**
- 模块导入测试
- 管道创建和运行
- T5 编码器预热
- 内存分配和释放

✅ **性能监控**
- 内存使用趋势
- 设备利用率
- 操作耗时分析
- 资源泄漏检测

## 💡 使用建议

1. **新环境部署**: 运行完整测试套件
2. **日常维护**: 使用快速测试
3. **问题诊断**: 针对性运行相关测试
4. **性能优化**: 使用监控工具分析
5. **服务启动**: 先运行快速检查

所有测试工具都支持 `--help` 参数查看详细使用说明。