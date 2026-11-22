# CoverDrive 项目说明

CoverDrive 是车险续保/加保的浏览器语音 Agent 演示。浏览器使用麦克风与 WebSocket，不连接 PSTN/SIP。

## 运行与配置

具体模型、服务地址及凭据由使用者在本机 `.env` 配置，详见 README.md 和 `.env.example`。仓库不提供 API Key、现成模型账号、保险 MCP 服务或客户数据。业务写入默认关闭；工具失败时失败关闭，不模拟报价或保单。

## 代码入口

- `backend/app/main.py`：FastAPI HTTP/实时 WebSocket 入口。
- `backend/voice_agent/`：原生语音协议、Deep Agents、工具授权和任务交付。
- `backend/app/`：兼容语音组件；`llm.py` 不是当前主 Agent 入口。
- `frontend/src/`、`frontend/public/`：页面、麦克风采集、PCM 播放、打断和任务状态。
- `backend/tests/`、`backend/tests_ksa/`、`frontend/tests/`：离线 fake 回归。
- `scripts/agent/`、`.agents/skills/product-delivery/`：本地交付校验。

## 验证

`npm run verify` 执行后端离线测试、前端测试与构建、Delivery 合同测试和任务校验。`npm run agent:candidate` 观察当前候选；Git checkout 使用 snapshot-git-v2，普通目录使用 snapshot-fs-v1。没有远端 CI、真实模型或真人语音验证承诺。

## 状态和候选

`tasks/`、`runs/` 为本地状态，不提交历史记录。候选排除真实秘密、这些状态、虚拟环境、Python 缓存、覆盖率、前端依赖/构建、日志、操作系统文件和本机配置；规则以 `scripts/agent/candidate-revision.mjs` 为准。产品源码、测试、配置示例、锁文件、VAD 资源和 Delivery 运行层必须参与候选。

## 边界

自动化使用 fake，不调用付费模型或真实业务系统。外部 MCP、模型、正式报价/出单、数据合规和生产部署由使用者另行配置并明确授权。本项目无生产鉴权、持久化会话或电话网关保证。新增旅程遵守 AGENTS.md 的 UX、最终转写和取消清理约束。
