# Voice Agent Agent-driven Delivery 规则

## 开始工作前

1. 阅读本文件、`PROJECT_PROFILE.md`、活动任务 `tasks/TASK-*.md`、相关产品/UX 文档和受影响代码。
2. 先确认任务目标、版本化验收项、基线、候选快照、写入边界、外部依赖和授权；事实冲突时以当前代码、测试和可复现命令为准并记录差异。
3. 先检查当前目录是否为 Git worktree；用 `scripts/agent/candidate-revision.mjs` 观察候选版本，不假设分支、提交或回滚已存在。
4. 保护现有文件，不覆盖或回滚无关工作。主 Agent 是唯一父任务编排者和状态所有者。

## 产品约束

- 产品是车险续保/加保外呼语音 Agent；浏览器“拨号”只是麦克风 + WebSocket 演示，不得宣称接通真实电话网络。
- 涉及金额、折扣、报价、保单或风险评分必须来自对应工具结果，不得心算、猜测或伪造。
- 外部工具失败或无数据时必须失败关闭，不得模拟客户、报价、保单或成功写入。
- 业务动作必须使用 ASR 最终文本；partial 只用于界面反馈，不能驱动报价、打标或出单。
- 客户明确同意前不得创建生效保单；明确拒绝后应停止推销。
- 保持端到端流式行为、工具调用前后分段、静默工具、打断取消和资源清理语义。修改这些路径必须覆盖正常、失败、超时和取消。
- `backend/app/llm.py` 当前不是主 Agent 路径；不要把它当成真实入口。

## Agent 分工与写入边界

- **main-agent**：唯一编排者；维护任务状态、集成变更、运行 `agent:check`、绑定 candidate、执行最终校验。可写任务、证据和已批准范围。
- **product_explorer**：只读定位用户流程、真实入口、数据流、测试和影响范围；不得修改文件。
- **ux_researcher**：只读检查通话流程、状态、文案、权限、打断、错误恢复、响应式和可访问性；E1/E2 不得冒充真人验证。
- **feature_worker**：每次只拥有一个模块：`backend/app/`、`frontend/src/` + `frontend/public/`，或明确的测试/文档范围。同一模块同时只允许一个生产代码写入者。
- **tester**：默认只写 `backend/tests/` 或新建的测试/fixture；不得修改 `backend/app/`、`frontend/src/`、`frontend/public/` 来让测试通过。
- **reviewer**：实现完成后只读审查正确性、隐私、外部写入、并发/取消、降级、可访问性和缺失测试。
- 共享目录只能并行相互独立的只读任务。并行写入必须先隔离目录、文件、端口、数据与外部账号，否则串行。
- 子 Agent 只返回 `completed`、`blocked` 或 `needs_decision` 及实际版本/命令/证据；其完成不会自动推进父任务状态。

## UX 闸门

- 新增或实质改变语音/文本旅程属于至少 L2 + `ux`：实施前记录目标用户、入口、成功终点、idle/listening/thinking/speaking/error 状态、麦克风拒绝、网络/供应商失败、打断恢复、目标浏览器与验证方法。
- 改变报价/出单、客户数据展示、权限请求或错误恢复必须进行 UX 走查；浏览器行为至少验证键盘可达、明确焦点、状态可辨、错误可恢复和窄屏布局。
- 自动化可以证明可观察行为，Agent 走查只能算 E1/E2；只有真实参与者记录才是 E3，线上真实数据才是 E4。
- 在没有可信来源适配器时，不得把 required `human_review`/`user_study` 作为 accepted 已满足；记录为非 required 或明确阻塞。

## 权限与数据边界

- 默认 autonomy 最多为 `implementation`：允许读取、修改授权文件和执行本地离线验证；不授予外部系统或生产权限。
- 未经明确授权不得：真实客户打标、报价/出单、飞书或保险系统写入、生产/预览发布、主分支合并、数据库迁移、凭据/域名/账单变更或付费资源创建。
- 不运行会打印敏感值或客户数据的命令。`.env` 和 `语音链路对接参数-完整配置.md` 只可按需确认键名或存在性，绝不复制值到任务、证据、日志或回复。
- 测试默认使用 fake/隔离环境；连接真实 LLM、ASR、TTS、MCP 前必须明确环境、数据和费用授权。

## 任务生命周期与证据

- 生命周期：`proposed → ready → in_progress → in_validation → accepted`；本项目尚未启用 `released`。
- `ready` 起必须有 baseline；`in_progress`/`in_validation` 必须有 active_run；`in_validation`/`accepted` 必须绑定当前 candidate。
- 合同字段或验收定义变化要增加任务 revision 并重跑受影响证据；纯状态迁移可保持 revision。
- 自动验收只通过任务内声明的精确 argv 执行：`npm run agent:check -- --task tasks/TASK-ID.md --criterion AC-ID --summary "..."`。
- 证据绑定 task revision、合同摘要、criterion 版本/摘要、candidate、argv 和产物摘要，位置为 `runs/TASK-ID/RUN-ID/`。
- `not_run`、`blocked`、`skipped`、旧 candidate、旧 revision、调用者替换命令或事后篡改产物都不算通过。
- accepted 前运行 `npm run agent:validate`。本地 `npm run agent:ci -- --require-replay` 只证明 fresh-run 机制，不能称为远端 CI。

## 候选规则

- 规则和回归测试分别在 `scripts/agent/candidate-revision.mjs` 与 `scripts/agent/agent-cli.test.mjs`；用 `npm run agent:candidate` 观察当前候选。
- 必须包含产品源码、测试、配置示例、依赖清单/锁文件、VAD 模型和 Delivery 运行层。
- 只可排除 `PROJECT_PROFILE.md` 已列出的 secrets/state/cache/build/log 路径；新增排除必须有“变化不改变 candidate”的回归，同时证明相邻源码/测试/配置/锁文件变化会改变 candidate。

## 验证命令

- 后端测试：`npm run test:backend`
- 前端构建：`npm run build:frontend`
- 产品验证：`npm run verify:product`
- Delivery 合同测试：`npm run agent:test`
- 当前候选：`npm run agent:candidate`
- 活动任务校验：`npm run agent:validate`
- 完整本地验证：`npm run verify`
- 本地非空重放：`npm run agent:ci -- --require-replay`

失败先分类为实现缺陷、环境失败、规范冲突或产品假设被证伪；只为第一类修改产品代码。不得弱化断言、门禁或验收标准来获得通过。
