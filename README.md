# CoverDrive

车险续保 / 加保场景的浏览器语音 Agent 演示。端到端语音处理连续对话与打断，Deep Agents 执行独立任务，通过可选 MCP 适配器查询真实业务结果。

> 浏览器“拨号”使用本机麦克风和 WebSocket，**不连接真实电话网络（PSTN/SIP）**。

## 先配置你自己的模型和 Key

**本仓库不提供 API Key、模型账号或已配置模型。首次运行前必须填写以下两组独立配置：**

| 用途 | 必须由你提供 |
| --- | --- |
| 端到端实时语音：连续对话、语音输入和输出 | `SEEDUPLEX_URL`、`SEEDUPLEX_MODEL`、`SEEDUPLEX_SPEAKER`、`SEEDUPLEX_API_KEY` |
| LLM：Deep Agents 任务推理、工具调用 | `LLM_BASE_URL`、`LLM_MODEL_ID`、`LLM_API_KEY` |

在项目根目录复制示例，再用编辑器填写自己的值：

```bash
cp .env.example .env
```

- `LLM_BASE_URL` 填支持 **OpenAI Chat Completions 流式输出与工具调用**的 API 基地址；`LLM_MODEL_ID` 填该服务支持的模型 ID。
- 当前端到端适配器实现 **Seeduplex 原生 JSON WebSocket 协议**，支持工具调用、16 kHz PCM 输入及 24 kHz PCM16 输出。请选择兼容此协议的模型、地址和音色，并确认账号已有对应权限。它不是通用的任意语音供应商适配器；更换协议需要实现新的 provider。
- 端到端语音 Key 与 LLM Key 分别申请、分别填写，即使两个服务来自同一平台也不要假设凭据可通用。模型费用由你与自己的服务商结算。
- 若使用语音服务的旧版应用凭据认证，可将 `SEEDUPLEX_API_KEY` 留空，填写你自己的 `SEEDUPLEX_APP_ID`、`SEEDUPLEX_ACCESS_KEY`、`SEEDUPLEX_APP_KEY` 和服务要求的 `SEEDUPLEX_RESOURCE_ID`。优先推荐 API Key 方式。
- `.env` 仅保存在本机，已加入 `.gitignore`。不要将 Key 写入前端、源码、截图、日志或提交记录。

缺少配置时，页面会提示填写本地 `.env`；语音及任务请求失败关闭。补齐配置后重启后端，再开启语音。`/api/health` 只报告缺少的变量名及模型角色，不返回 Key 或具体模型 ID。

## 本地启动

需要 Python 3.11+、Node.js 22.12+ 和 npm。浏览器推荐近期 Chrome / Edge，麦克风需 localhost 或 HTTPS 安全上下文。

```bash
git clone https://github.com/IcecreamAgent/CoverDrive.git
cd CoverDrive

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm --prefix frontend ci

cp .env.example .env
# 用编辑器填写上面的端到端语音和 LLM 配置
```

终端 1（项目根目录）：

```bash
npm run dev:backend
```

终端 2（项目根目录）：

```bash
npm run dev:frontend
```

打开 Vite 输出的本地地址（默认 `http://localhost:5173`），允许麦克风后开启语音。键盘输入直接进入任务 LLM；只使用文字时也必须配置 LLM 地址、模型与 Key。

可检查后端状态：

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

`ok: true` 表示进程可用；还需 `configured: true` 才代表必填配置齐全。配置检查不验证账号授权、余额、模型可用性或网络。供应商失败时界面保留错误反馈与重试入口，不会伪造成功结果。

## 保险业务工具

保险 MCP 是外部依赖，**仓库不附带保险服务、客户数据库或现成报价**。示例默认 `INSURANCE_TOOLS_ENABLED=false`、`INSURANCE_WRITES_ENABLED=false`，建议先用键盘输入验证 LLM 任务。

需要选客、查询、核价或保单相关演示时，先自行部署兼容的 MCP 服务，再填写 `INSURANCE_MCP_URL` 并开启 `INSURANCE_TOOLS_ENABLED=true`。工具不可用或无数据时会失败关闭；金额、折扣、报价、保单和风险结果必须来自工具，不使用模拟结果。当前页面的拨号按钮开启语音后会尝试查询一位续保客户。未配置 MCP 或关闭保险工具时，会显示“无法查询可联系客户”的错误，不会生成客户或报价；语音连接保留，你仍可继续说话。完整的选客与销售开场演示必须先配置可用的 MCP；修复后清空对话并重新拨号。

业务写入默认关闭。启用前须自行完成鉴权、数据授权与测试，客户明确同意后才可出单；明确拒绝后停止推销。浏览器拨号不授权业务写入。

## 安全与部署边界

- Key 只由后端读取，前端仅通过同源 `/api`、`/ws` 访问后端。仓库不包含作者的私密配置、运行记录或具体部署模型。
- 本地默认无访问令牌，仅监听回环地址。任何非本机使用前配置 `VOICE_AUTH_TOKEN`（页面提供令牌输入），并配置 HTTPS、`ALLOWED_ORIGINS` 和 `FRONTEND_ORIGIN`，另外完成部署与安全评估。访问令牌不是服务商 API Key。
- 不建议直接公开开发服务器。本演示不提供生产用户系统、持久化多实例会话、真实电话网关或生产部署保证。
- 历史兼容 ASR/TTS 路径保留在源码中；当前默认原生端到端语音不需要单独 ASR/TTS Key。

## 离线验证

安装依赖后在根目录运行：

```bash
npm run verify
node scripts/check-public.mjs
```

测试使用 fake / 隔离 fixture，不需要真实模型 Key，不调用付费模型、真实客户或业务系统。`verify` 包括后端、前端、构建及本地交付合同测试；这些通过不代表真实模型兼容性、真人语音体验或远端 CI 已验证。

## 代码入口

- `backend/app/main.py`：FastAPI HTTP / WebSocket。
- `backend/voice_agent/`：原生语音协议、Deep Agents、独立任务、工具授权、打断与交付。
- `frontend/src/`、`frontend/public/`：界面、音频采集与播放。
- `backend/tests/`、`backend/tests_ksa/`、`frontend/tests/`：离线测试。
- `AGENTS.md`、`PROJECT_PROFILE.md`、`scripts/agent/`：项目约束与本地交付工具。

## 许可证

本项目采用 [MIT](LICENSE) 许可证。第三方依赖和模型资源遵循各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
