"""Selected-customer demo issuance uses the existing transaction/receipt path."""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from voice_agent.config import Settings
from voice_agent.insurance import InsuranceAuthorization, WRITE_TOOLS, guarded_tools


CUSTOMER = {"id": 91, "姓名": "隔离客户", "电话": "138****0091", "车牌号": "测T00091"}
IDENTITY = {"customer_name": CUSTOMER["姓名"], "customer_phone": CUSTOMER["电话"],
            "plate": CUSTOMER["车牌号"]}
POLICY = {"policy_no": "POL-ISOLATED-91", "status": "已承保", "total_premium": 4254.49}


class Arguments(BaseModel):
    customer_name: str = ""
    customer_phone: str = ""
    plate: str = ""
    status: str = "已生效"


def authorization(*, final="对，直接下单。", selected=True):
    auth = InsuranceAuthorization()
    if selected:
        auth.selected_customer = CUSTOMER.copy()
    if final is not None:
        auth.observe_final(final)
    return auth


def protected(auth, calls, *, name="create_policy", enabled=True,
              scope="selected_customer_issue", result=POLICY, failure=None, started=None):
    async def execute(**arguments):
        calls.append(arguments)
        if started:
            started.set()
            await asyncio.Event().wait()
        if failure:
            raise failure
        return result
    original = StructuredTool.from_function(coroutine=execute, name=name,
        description="Isolated fixture with no network or customer writes.", args_schema=Arguments)
    settings = SimpleNamespace(insurance_writes_enabled=enabled, insurance_write_scope=scope)
    return guarded_tools([original], settings, auth)[0]


def test_config_keeps_local_default_and_rejects_unknown_write_scope():
    assert Settings(_env_file=None).insurance_write_scope == "all"
    assert Settings(_env_file=None, insurance_write_scope="selected_customer_issue").insurance_write_scope == "selected_customer_issue"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, insurance_write_scope="invalid-fixture")


async def test_disabled_issuance_records_specific_unsubmitted_receipt_instead_of_generic_failure():
    auth, calls = authorization(), []
    result = await protected(auth, calls, enabled=False).ainvoke(IDENTITY)
    assert "未执行" in result and calls == []
    assert auth.policy_attempts == {auth.version}
    assert auth.policy_results[auth.version] == {"outcome": "not_submitted", "reason": "writes_disabled"}
    answer = auth.policy_confirmation_result(auth.version)
    assert "尚未开放出单" in answer and "未提交订单" in answer
    assert "没有取得有效保单，请核对后再办理" not in answer


@pytest.mark.parametrize("name", sorted(WRITE_TOOLS - {"create_policy"}))
async def test_demo_scope_does_not_open_unrelated_business_writes(name):
    auth, calls = authorization(), []
    result = await protected(auth, calls, name=name).ainvoke(IDENTITY)
    assert "不受支持" in result and "未执行" in result and calls == []
    if name == "renew_policy":
        assert auth.policy_results[auth.version] == {"outcome": "not_submitted", "reason": "write_scope"}
        assert "办理范围" in auth.policy_confirmation_result(auth.version)
    else:
        assert auth.policy_results == {}


@pytest.mark.parametrize("selected", [False, True])
async def test_demo_scope_requires_current_final_input_even_with_selected_customer(selected):
    auth, calls = authorization(final=None, selected=selected), []
    result = await protected(auth, calls).ainvoke(IDENTITY)
    assert "最终输入" in result and calls == []
    assert auth.policy_results[0] == {"outcome": "not_submitted", "reason": "final_input"}
    assert "未提交订单" in auth.policy_confirmation_result(0)


async def test_demo_scope_requires_selected_customer_and_returns_specific_receipt():
    auth, calls = authorization(selected=False), []
    result = await protected(auth, calls).ainvoke(IDENTITY)
    assert "尚未选定" in result and calls == []
    assert auth.policy_results[auth.version]["reason"] == "selected_customer"
    assert "尚未选定" in auth.policy_confirmation_result(auth.version)


@pytest.mark.parametrize("field", list(IDENTITY))
async def test_demo_scope_uses_exact_selected_identity_and_allows_pre_submission_correction(field):
    auth, calls = authorization(), []
    tool = protected(auth, calls)
    result = await tool.ainvoke({**IDENTITY, field: "其他隔离资料"})
    assert "不一致" in result and calls == []
    assert auth.policy_results[auth.version] == {"outcome": "not_submitted", "reason": "customer_identity"}
    assert "资料不一致" in auth.policy_confirmation_result(auth.version)
    await tool.ainvoke(IDENTITY)
    assert len(calls) == 1 and POLICY["policy_no"] in auth.policy_confirmation_result(auth.version)


async def test_demo_scope_passes_the_existing_tool_receipt_without_fabricating_success():
    auth, calls = authorization(), []
    tool = protected(auth, calls)
    result = await tool.ainvoke({"type": "tool_call", "id": "actual-demo-call",
        "name": "create_policy", "args": IDENTITY})
    assert isinstance(result, ToolMessage) and result.status == "success"
    assert len(calls) == 1 and auth.policy_results[auth.version]["outcome"] == "confirmed"
    assert auth.policy_results[auth.version]["customer_id"] == CUSTOMER["id"]
    answer = auth.policy_confirmation_result(auth.version)
    assert POLICY["policy_no"] in answer and "4254.49元" in answer
    assert "不重复办理" in await tool.ainvoke(IDENTITY) and len(calls) == 1


async def test_local_all_scope_still_allows_existing_unselected_tool_flow():
    auth, calls = authorization(selected=False), []
    await protected(auth, calls, scope="all").ainvoke(IDENTITY)
    assert len(calls) == 1 and POLICY["policy_no"] in auth.policy_confirmation_result(auth.version)


async def test_stale_task_cannot_borrow_a_new_final_input_in_demo_scope():
    auth, calls = authorization(), []
    old_version = auth.version
    tool = protected(auth, calls)
    auth.observe_final("先不要办理。")
    result = await tool.ainvoke(IDENTITY)
    assert "版本已过期" in result and calls == []
    assert auth.policy_results[old_version]["reason"] == "final_input"
    assert auth.policy_results.get(auth.version) is None


@pytest.mark.parametrize("failure", [TimeoutError(), RuntimeError("isolated unknown write")])
async def test_demo_scope_keeps_unknown_write_reservation_and_never_retries(failure):
    auth, calls = authorization(), []
    tool = protected(auth, calls, failure=failure)
    with pytest.raises(type(failure)):
        await tool.ainvoke(IDENTITY)
    assert auth.policy_results[auth.version] == {"outcome": "unknown"}
    assert "无法确认" in auth.policy_confirmation_result(auth.version)
    assert "不重复办理" in await tool.ainvoke(IDENTITY) and len(calls) == 1
    # A later preflight rejection may not overwrite a possibly committed write.
    denied = protected(auth, calls, enabled=False)
    await denied.ainvoke(IDENTITY)
    assert auth.policy_results[auth.version] == {"outcome": "unknown"}
    assert "无法确认" in auth.policy_confirmation_result(auth.version)


async def test_demo_scope_cancellation_remains_unknown_and_cannot_submit_twice():
    auth, calls, started = authorization(), [], asyncio.Event()
    tool = protected(auth, calls, started=started)
    task = asyncio.create_task(tool.ainvoke(IDENTITY))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert auth.policy_results[auth.version] == {"outcome": "unknown"}
    assert "不重复办理" in await tool.ainvoke(IDENTITY) and len(calls) == 1


async def test_known_business_error_does_not_become_a_receipt_or_a_retry_permission():
    auth, calls = authorization(), []
    tool = protected(auth, calls, result={"error": "isolated private detail"})
    result = await tool.ainvoke(IDENTITY)
    assert "没有有效结果" in result and "isolated private detail" not in result
    assert auth.policy_results[auth.version] == {"outcome": "failed"}
    assert "不重复办理" in await tool.ainvoke(IDENTITY) and len(calls) == 1
