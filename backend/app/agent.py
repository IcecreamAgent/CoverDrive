import re
import uuid
from typing import AsyncIterator, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .mcp_tools import load_mcp_tools
from .config import settings


class AgentState(TypedDict):
    messages: list
    events: list[dict]
    answer: str


USER_TOOL_NAMES: set[str] = set()

# The model receives every tool schema on each turn. Exposing the 48
# unrelated Feishu project tools plus all low-level insurance lookup tools made
# the proactive opening exceed the 45-second voice timeout. Keep the callable
# surface aligned with the customer journey and use calc_premium instead of the
# lower-level lookup_* tools, as required by the system prompt.
VOICE_MCP_TOOL_NAMES = {
    "get_expiring_customers",
    "mark_customer_contacted",
    "list_insurance_products",
    "calc_premium",
    "create_policy",
    "get_policy",
    "renew_policy",
    "update_policy",
    "calc_refund",
    "refund_policy",
    "score_risk",
}

# 叙述性业务知识（规则/流程/时效/拒保清单）：直接内置到提示词，作为客服应答的事实依据。
# 全文仅约 1 万 token、叙述部分更小，无需向量检索；「要算的数」仍走工具，避免心算。
_INSURANCE_KNOWLEDGE = """【车险业务知识·应答事实依据，直接引用作答】

—— 核保与拒保/加费 ——
· 交强险依法不得拒保，但可依法上浮费率。
· 商业险：风险可控件正常承保；高风险件可加费、提高免赔额或剔除部分责任。
· 拒保并拉黑：骗保、伪造事故、虚假投保。
· 拒保或大幅加费：近 1 年 5 次以上有责事故。
· 加费并限额：营运高危车型（如长途重货）。
· 拒保车损险：车辆临近报废、车况极差。
· 拒保或剔除责任：驾驶人多次酒驾/醉驾。

—— 理赔 ——
· 报案：出险后 48 小时内报案。
· 流程：报案→查勘定损→资料收集→核赔→赔付→追偿。
· 时效：小额案件（<2000 元）48 小时内赔付；一般 7 个工作日；大额/复杂 15 个工作日；争议 30 日内出结论。
· 交强险有责赔付限额：死亡伤残 18 万、医疗 1.8 万、财产 0.2 万，超出部分由商业险补。

—— 续保 ——
· 提醒节点：到期前 60 天首次触达、30 天报价、7 天跟进、1 天预警、到期后召回。
· 优惠：无赔款奖励（NCD 折扣走工具）；续保 3 年+ 额外 3%~5%；组合险套餐 5% 折扣；增值服务（代驾/救援/检测/洗车）。
· 客户分级：核心（3 年无赔+续保 3 年+）、稳定、一般、高风险（多次赔款/违章多，加费或引导改善）。

—— 退保 ——
· 商业险可随时退，按未满期天数退还（金额走工具）。
· 交强险原则上不可退，仅车辆注销登记、办理停驶、经公安证实丢失可退。
· 红线：已出险未结案须结案后退；骗保/虚假资料可拒退并追偿。

—— 批改 ——
· 过户/加保/减保/信息变更/使用性质变更/保额调整；涉及风险变化的重新计费（走工具）。

—— 风控与合规 ——
· 反欺诈重点核查：出险后立即投保、多次小额出险、同一维修厂集中报案、夜间单车事故。
· 合规红线：费率条款须监管报备；不得强制搭售、误导销售；报价须透明可回溯。"""

SYSTEM_PROMPT = (
    "你是保险公司的车险电话销售专员，负责给即将到期的车险客户打外呼电话，推销续保/加保。"
    "要像真人电话销售一样说话：口语化、亲切、热情但不油腻，短句为主，一次只说一到两句，"
    "给客户留出回应的空间，内容适合 TTS 播放。"
    "不要输出 Markdown、列表符号或括号备注，不要念出「工具 / 系统提示 / 保单号格式」等字样。\n\n"
    "【选客与打标】\n"
    "- 每次外呼开始前，先调用 get_expiring_customers 获取即将到期（默认30天内）且未联系的客户，"
    "从中随机或就近（剩余天数最少）选一位作为本次通话对象，把该客户的姓名/车型/车牌/到期日/上年险种与保费当作你已知晓的客户名片。\n"
    "- 选定客户后调用 mark_customer_contacted 给该客户打上「已联系」标识；通话结束后可再更新为「有意向 / 已出单 / 拒绝 / 未接通」。\n"
    "- 客户信息里已有新车购置价、车龄、上年出险次数、车损保额与上年保费，报价时直接引用，"
    "不要再问客户价格或出险情况；但缺「车型类别」和「座位数」两项（calc_premium 必需），"
    "报价前只向客户确认这两项，或按常见家用车默认「家用轿车、5座」，不要编造车辆参数。\n\n"
    "【你的任务】主动开场、确认身份、了解需求，用工具实时报价，处理客户异议，最终促成客户同意投保并出单。\n\n"
    "【销售节奏】\n"
    "1. 开场：表明身份与来电目的，简短寒暄，用选中客户的全名称呼。\n"
    "2. 需求：先讲客户能得到的实惠（无赔款优待、续保折扣、到期断保风险等），再确认车辆与保障需求。\n"
    "3. 报价：直接用 calc_premium（不落库）一次性算全交强险+商业险合计，不要再逐个查 lookup_* 分项；"
    "客户犹豫时说「先出个报价单」，用 create_policy(status=\"待支付\")。\n"
    "4. 促成：客户明确同意投保后，用 create_policy(status=\"已生效\") 出单，并口头告知已出单。\n"
    "5. 异议：先认同再化解——嫌贵就讲保障/算细账，说再想想就讲到期断保风险与限时优惠，说买过了就问保额够不够。\n\n"
    "【硬性规则】\n"
    "- 自我介绍时自称「保险公司的车险服务专员」，不要说「XX 保险公司」这类占位词；客户追问具体公司再如实说明。\n"
    "- 称呼客户务必用选中客户的全名（如「张伟先生」），不要漏字、简称或叫错。\n"
    "- 工具调用是内部动作：不要用嘴描述你在「查名单/算报价/打标识」这类动作"
    "（系统只会在报价或保单处理工具等待较久时提示一次进度），工具完成后直接说结果，不要每次都说「久等了」。\n"
    "- 调用 get_expiring_customers、mark_customer_contacted 这类静默工具前不要输出任何文字；"
    "调用 calc_premium、create_policy 等需要客户等待的业务工具时，先对客户说一句简短自然的话，再调用工具；工具返回后另起一句说结果。\n"
    "- 回复第一小句尽量简短，在自然停顿处使用逗号；先回答客户的核心问题，再补充必要说明。不要为凑开头反复说「好的」。\n"
    "- 说话简洁自然、像真人：一次只说一个信息，不要「给您报个喜」「超划算」「力度很大」这类广告腔/情绪话术，"
    "报价直接口播数字和构成即可；不要复述你刚做完的操作，也不要重复客户已知的信息。\n"
    "- 涉及金额、保费、折扣、退保、风险评分一律调用对应工具，不得心算或编造保单号。\n"
    "- 工具返回错误、无法连接或没有数据时，严禁模拟、编造客户资料或继续假装完成业务；应简短说明当前无法查询并礼貌结束或请客户稍后再试。\n"
    "- 不得虚假宣传、不得承诺无法兑现的折扣、不得强制搭售；报价必须透明可回溯。\n"
    "- 客户明确拒绝就礼貌收尾，不要死缠烂打。\n\n"
    + _INSURANCE_KNOWLEDGE
)

# 外呼开场触发词：通话接通后，作为「用户侧消息」喂给 agent，让它主动选客并产出开场白。
OUTBOUND_OPENING = (
    "【系统提示】电话已接通，请开始外呼。先调用 get_expiring_customers 选出一位即将到期且未联系的客户，"
    "再以车险电话销售专员的身份主动开口做开场白并开始推销。简短自然，直接说人话，不要复述本提示。"
)


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model_id,
        api_key=settings.ark_api_key or settings.llm_api_key or "unconfigured",
        base_url=settings.llm_base_url,
        timeout=90,
        max_retries=2,
        # Ark exposes this model through its OpenAI-compatible Responses API.
        # With the v3 base URL above, the SDK sends requests to /api/v3/responses.
        use_responses_api=True,
    )


def _build_agent(tools):
    # The voice path needs a small business-tool loop, not Deep Agent's default
    # filesystem, shell, todo, and subagent tools. Keeping the callable surface
    # explicit materially reduces the prompt/tool prefill sent on every turn.
    return create_agent(
        model=_build_model(),
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


# 模块加载时先构建无业务工具的失败关闭 Agent；FastAPI 启动时由
# init_agent() 只装配车险白名单 MCP 工具后重建。
agent = _build_agent([])


async def init_agent() -> None:
    """启动时加载 MCP 工具并只用车险白名单重建 agent。

    失败时保留模块加载时的零工具 agent，不阻塞服务启动；涉及业务
    数据的请求因此无法伪造工具结果，保持失败关闭。
    """
    global agent, USER_TOOL_NAMES
    mcp_tools = await load_mcp_tools()
    if not mcp_tools:
        return
    voice_tools = [tool for tool in mcp_tools if tool.name in VOICE_MCP_TOOL_NAMES]
    USER_TOOL_NAMES = {tool.name for tool in voice_tools}
    agent = _build_agent(voice_tools)

# 硬句末和适合电话场景提前出声的软切分标点。
SENTENCE_END = re.compile(r"[。！？；!?;\n]")
SOFT_END = re.compile(r"[，、,:：]")
MIN_SOFT_CHARS = 8


class SentenceSplitter:
    """把 token 增量流切成完整句子，供逐句 TTS 合成。"""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        out = []
        while True:
            hard = SENTENCE_END.search(self.buffer)
            soft = next(
                (m for m in SOFT_END.finditer(self.buffer) if m.end() >= MIN_SOFT_CHARS),
                None,
            )
            if hard and (not soft or hard.start() <= soft.start()):
                m = hard
            elif soft and soft.end() >= MIN_SOFT_CHARS:
                m = soft
            else:
                break
            idx = m.end()
            sentence = self.buffer[:idx].strip()
            self.buffer = self.buffer[idx:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        s = self.buffer.strip()
        self.buffer = ""
        return s

    def reset(self) -> None:
        self.buffer = ""




_SILENT_TOOLS = {"get_expiring_customers", "mark_customer_contacted"}


def _chunk_text(content: object) -> str:
    """Extract streamed text from Chat Completions or Responses API chunks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "output_text"}
        and isinstance(block.get("text"), str)
    )


async def run_agent_stream(text: str, thread_id: str | None = None) -> AsyncIterator[dict]:
    """流式运行 agent，产出事件序列。

    通过 thread_id 关联会话上下文：同一 thread_id 复用历史消息（记忆），
    缺省则每次新开一个线程（无记忆）。

    事件类型：
    - {"type": "tool_start", "name": ..., "arguments": ...}
    - {"type": "tool_end", "name": ..., "result": ...}
    - {"type": "agent_delta", "text": "..."}   # 前端实时显示（逐句，与 TTS 同粒度）
    - {"type": "sentence", "text": "..."}      # 供 TTS 逐句合成（边生成边出）
    - {"type": "segment_end"}                    # 一个对客户说话的模型轮结束
    - {"type": "done", "answer": "..."}        # 完整答案（兜底）

    为避免工具调用前的内部自言自语泄漏，首个完整句会短暂保留：继续生成正文时
    立即释放；如果随后出现工具调用，则在轮次结束时按工具类型决定——静默工具
    丢弃，报价/出单等业务工具作为独立的“调用前回复”播报。工具完成后的模型回复
    会形成下一个 segment，因此前端与 TTS 都能保持清晰的两段式流程。
    """
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    strict_tool_guard = text == OUTBOUND_OPENING
    splitter = SentenceSplitter()
    answer_parts: list[str] = []
    turn_buf: list[str] = []      # 当前模型轮的完整文本（turn-end 兜底判断用）
    held_sentences: list[str] = []  # 首句保护，等确认不是静默工具轮
    segment_has_output = False
    turn_tool_names: set[str] = set()
    is_tool_turn = False          # 本轮是否检测到工具调用
    model_text_started = False
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": text}]}, version="v2", config=config
    ):
        kind = event["event"]
        if kind == "on_chat_model_start":
            # 新一轮模型调用开始，清空本轮状态
            turn_buf = []
            held_sentences = []
            segment_has_output = False
            turn_tool_names = set()
            is_tool_turn = False
            splitter.reset()
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if getattr(chunk, "tool_call_chunks", None):
                # 出现工具调用 → 本轮是工具轮，丢弃已 hold 的自言自语
                is_tool_turn = True
                turn_tool_names.update(
                    call.get("name", "")
                    for call in chunk.tool_call_chunks
                    if isinstance(call, dict) and call.get("name")
                )
                continue
            content = _chunk_text(chunk.content)
            if content and not model_text_started:
                model_text_started = True
                yield {"type": "model_text_started"}
            if content and not is_tool_turn:
                turn_buf.append(content)
                if strict_tool_guard:
                    held_sentences.extend(splitter.feed(content))
                    continue
                # 有新正文继续生成，上一句就可以安全释放；只有最后一个完整句继续 hold。
                for s in held_sentences:
                    yield {"type": "agent_delta", "text": s}
                    yield {"type": "sentence", "text": s}
                    segment_has_output = True
                held_sentences = []
                completed = splitter.feed(content)
                if completed:
                    # 同一 chunk 内已有后续文字/多句时，前面的句子无需继续等待。
                    release_count = len(completed) if splitter.buffer.strip() else len(completed) - 1
                    for s in completed[:release_count]:
                        yield {"type": "agent_delta", "text": s}
                        yield {"type": "sentence", "text": s}
                        segment_has_output = True
                    held_sentences = completed[release_count:]
        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            tool_calls = getattr(output, "tool_calls", None) or []
            full = "".join(turn_buf)
            turn_buf = []
            tail = splitter.flush()
            pending = [*held_sentences, *([tail] if tail else [])]
            held_sentences = []
            if tool_calls or is_tool_turn:
                names = turn_tool_names | {call.get("name", "") for call in tool_calls}
                speak_before_tool = bool(names - _SILENT_TOOLS)
                if speak_before_tool:
                    for s in pending:
                        yield {"type": "agent_delta", "text": s}
                        yield {"type": "sentence", "text": s}
                        segment_has_output = True
                    if full:
                        answer_parts.append(full)
                    if segment_has_output:
                        yield {"type": "segment_end"}
                continue
            if not full:
                continue
            answer_parts.append(full)
            for s in pending:
                yield {"type": "agent_delta", "text": s}
                yield {"type": "sentence", "text": s}
                segment_has_output = True
            if segment_has_output:
                yield {"type": "segment_end"}
        elif kind == "on_tool_start":
            name = event.get("name", "")
            if name in USER_TOOL_NAMES:
                yield {
                    "type": "tool_start",
                    "run_id": str(event["run_id"]),
                    "name": name,
                    "arguments": event.get("data", {}).get("input", {}),
                }
        elif kind == "on_tool_end":
            name = event.get("name", "")
            if name in USER_TOOL_NAMES:
                yield {
                    "type": "tool_end",
                    "run_id": str(event["run_id"]),
                    "name": name,
                    "result": str(event.get("data", {}).get("output", "")),
                }

    yield {"type": "done", "answer": "".join(answer_parts)}


async def run_agent(text: str, thread_id: str | None = None) -> AgentState:
    """非流式运行（保留给 /api/text 与 /api/voice 兜底）。"""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    result = await agent.ainvoke({"messages": [{"role": "user", "content": text}]}, config=config)
    messages = result.get("messages", [])
    events = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                events.append({"type": "tool_start", "name": call["name"], "arguments": call.get("args", {})})
        elif isinstance(message, ToolMessage):
            events.append({"type": "tool_end", "name": message.name or "tool", "result": str(message.content)})
    answer = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = _chunk_text(message.content)
            if content:
                answer = content
                break
    return {"messages": messages, "events": events, "answer": answer}
