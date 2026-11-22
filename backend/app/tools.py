from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import StructuredTool


def get_current_time(city: str = "上海") -> str:
    zones = {"上海": "Asia/Shanghai", "北京": "Asia/Shanghai", "东京": "Asia/Tokyo", "纽约": "America/New_York"}
    zone = zones.get(city, "Asia/Shanghai")
    return datetime.now(ZoneInfo(zone)).strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not expression or any(c not in allowed for c in expression):
        return "只允许进行基础四则运算。"
    try:
        return str(round(eval(expression, {"__builtins__": {}}, {}), 8))
    except Exception:
        return "无法计算这个表达式。"


def create_tools():
    return [
        StructuredTool.from_function(get_current_time, name="get_current_time", description="查询指定城市当前时间。"),
        StructuredTool.from_function(calculate, name="calculate", description="计算基础四则运算表达式。"),
    ]
