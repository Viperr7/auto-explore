# Verify 驱动 Auto Explore：定点补采 v1

补采不是重跑全图。评估报告只产生候选；独立复核 A 确认真缺口并限定路线；Auto 只执行通过实时状态和安全校验的目标。只有验证成功的目标边进入新图。复评用同一份 SRM、同一评估配置；用补采前冻结的独立参考 B 对比页面和路径。现有评分和报告格式不变。

## 1. 离线生成候选

在 Auto 仓库下执行。`--output` 必须是新文件；其父目录应已存在。此命令不连接设备。

```powershell
Set-Location -LiteralPath 'D:\华为诺亚轨迹合成项目\auto-explore'
$py = 'C:\Python314\python.exe'
$old = 'D:\华为诺亚轨迹合成项目\outputs\stg-report-ui-20261118\bohe-final'
$out = 'D:\华为诺亚轨迹合成项目\outputs\stg-verify-repair-20261118\bohe-dry-run'
& $py -B -X utf8 -m harness.repair_plan `
  --stg 'D:\stg\bohe_first_round' `
  --report "$old\stg_quality_report.json" --gaps "$old\closure_gaps.json" `
  --version-evidence 'D:\stg\bohe_first_round\states\round_001\package.txt' `
  --environment bohe-15.10.1-archive --output "$out\repair_plan_repro.json"
```

`repair_plan.json` 分 `executable` / `review` / `skip`。不传 `--review` 就没有可执行任务。不要手改 `jobs` 批准动作。静态误对齐、疑似错拆分应先修映射，不派采样任务。

### 独立复核 A 的输入

见 [可控样例的复核文件](D:/华为诺亚轨迹合成项目/outputs/stg-verify-repair-20261118/controlled-final/verify_A_review.json)。把评审人认可的复核文件作为 `--review` 输入，重新生成一份新计划：

- 复核文件 `format=stg_repair_review`、`schema_version=1`。包名、`[versionName, versionCode]`、环境标识及报告、基线、缺口、原始图哈希须与离线清单一致。
- A 是未参与构图的同版本 `app_graph.json`，要有真实的冷启动任务起点、连续 `edge_ids`、成功动作和前后截图／控件树。`trace` 和 `version_evidence` 各含绝对 `path` 与 `sha256`；`asset_sha256` 用相对 A 图目录的 POSIX 路径绑定审批时的截图和原生 XML。不要用基线图或拷贝其截图充当 A。
- 每个 `verified_missing` case 要关联 `candidate_id`、`baseline_observation`，给出人工复核原因、确认过的 `route` 和一个 `target`。每步包含原始 `edge_id`、`read_only_navigation` 或 `read_only_expansion`、完整资源 ID 和精确标签选择器。旧点击坐标只用于回查，不作为执行坐标。不同账号／版本／条件下不能宣称同一状态；`--environment` 是操作者声明，不自动识别账号。
- B 独立采集并在补采前冻结，只交验收器，不交补采生成器、Auto 或合并器。

## 2. 定点执行与证据导出

仅当同版本测试设备、测试账号和动作已获授权时，才使用真实 `--device`。`$plan`、`$environment`、`$package`、`$serial`、`$run` 均为操作者指定；`$run` 必须不存在。

```powershell
& $py -B -X utf8 .\explore_android_app.py --package $package --device $serial `
  --repair-environment $environment --repair-plan $plan --output $run
```

显式定点模式不走常规漫游、不调用模型。默认最多5个任务、40次动作、30分钟；重启中的停止和启动、恢复过程及失败尝试均计入。目标动作一般仅执行1次，只有复核允许的无反馈任务可额外执行1次。跨包、画面过期、控件歧义、缺少原生控件树、危险操作及未验证分支不会产出成功边。

- `repair_results.json`：逐项结果、拒绝或停止原因、动作次数、路线和证据引用。
- `repair_observations.json` 与 `assets/`：实际截图、原生控件树；仅截图的见证保留，但不能当成已验证的控件证据。
- `app_graph.json`：仅成功目标与真实端点。重启、恢复和未确认结果不构成补采业务边。

## 3. 合并到新目录，再用原公式复评

```powershell
& $py -B -X utf8 -m harness.repair_merge --plan $plan --repair-run $run `
  --baseline $baseline --output $newTopology
```

当前仅支持 Research 的 `topology/data.json` 基线及可回查的源图；语义合并 `merge_report.json` 暂不接入，不能降级后强行合并。适配器复用原 `write_topology`，保留基线源图、节点映射和边，并用原评估读取器核验端点。只在有已验证新增时生成新图。图的评估入口依赖 `run_manifest.json` 引用的原始路径，归档时应连同基线源图和补采证据一起保留。

再用现有 STG-eval 对旧图／新图以同一份 SRM 和配置复评，输出分别放到新目录；用同一冻结 B 单列页面及路径结果。本例复用的是 `gym_explore.evaluation.evaluate.evaluate` 与 `visible_coverage.evaluate_visible`，不改变二者实现。小样例可这样复现，不连接设备：

```powershell
& $py -B -X utf8 .\tests\test_repair_closed_loop.py `
  'D:\华为诺亚轨迹合成项目\outputs\stg-verify-repair-20261118\controlled-repro'
```

## 边界

首版宁可拒绝，也不从标签、模型描述或分数猜动作。动态画面、WebView／自绘控件、UIAutomator 拿不到稳定层级、状态只能靠视觉区分、相同按钮对应不同业务条件等情形，需要独立复核；不能视为覆盖。此次只验证了离线和可控模拟设备，不代表已经验证真实 Android ADB 后端或薄荷健康正式补采。
