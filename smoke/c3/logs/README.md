# 完整日志与采集规范

## 已完成的运行

```text
smoke/c3/logs/
├── preflight/environment.txt
├── *_gpu_fp32_seed824/   # 三个稳定 GPU 运行
├── *_gpu_seed824/        # 三个 AMP 降级运行
└── *_cpu_seed824/        # 三个 CPU 功能运行
```

9 次训练均正常退出。GPU FP32 组三次运行没有自动数值恢复；GPU AMP 组三次运行都触发一次非有限值恢复，训练器关闭 AMP 后重试，因此标记为 `DEGRADED`；CPU 组只用于辅助功能核验。统一结果说明见 `smoke/c3/docs/ADMISSION_20260825.md`。

## 每个运行目录的文件

| 文件 | 内容 |
| --- | --- |
| `command.txt` | 不含令牌或凭据的实际命令 |
| `train.log` | 从启动到最终验证结束的完整 stdout/stderr 行 |
| `resolved_config.yaml` | YOLO 最终解析配置 |
| `metrics.csv` | 每轮训练损失与验证指标 |
| `resources.csv` | UTC 时间、CPU 使用率、RSS、GPU 编号与 GPU 显存的每秒采样 |
| `time.txt` | 开始/结束 UTC、总耗时和退出码 |
| `result.json` | 状态、资源摘要、运行路径与产物信息 |
| `artifact_index.txt` | 本地产物路径、大小和 SHA-256 |

V-PEFT 目录还包含：

- `adapter_config.json`：独立 PEFT 适配器配置。
- `vpeft_runtime_metadata.json`：放置方案、目标审计、安全覆盖和实际后端。
- `postprocess.txt`：独立适配器导出状态。

## 路径与完整性策略

可提交日志保留所有输出行并移除 ANSI 控制符，只把仓库根目录和用户主目录前缀替换为 `<repo>` 与 `<user-home>`。这样既保留故障上下文和完整进度，又不会上传本机用户绝对路径。原始训练产物保留在仓库忽略的 `runs/` 下，`artifact_index.txt` 使用 SHA-256 将证据与本地文件绑定。

运行器拒绝覆盖同名日志和训练目录。失败运行同样保留 `train.log`、资源采样、退出码和已生成产物，不会被成功运行覆盖。

`result.json` 的 `numerical_recovery` 通过“实际 epoch 尝试数减去配置 epochs”识别自动重试，同时记录 AMP 是否因恢复被关闭。GPU 显存来自每秒一次的 GPU 设备级采样，不写成精确的进程独占峰值。
