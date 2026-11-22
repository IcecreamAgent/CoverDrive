"""Conservative local evidence that a completed answer did not settle a topic.

This does not classify general task complexity. It uses recent public Q&A only,
never incomplete ASR revisions or interrupted answers. Lexical overlap cannot
recognize every paraphrase; the native model's semantic instructions complement
these deterministic cues rather than treating the score as semantic certainty.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

FollowupReason = Literal["none", "repeated_question", "dissatisfied_followup", "topic_followup"]


@dataclass(frozen=True, slots=True)
class FollowupDecision:
    escalate: bool
    reason: FollowupReason
    related_history: list[tuple[str, str]]


_HARMLESS = re.compile(
    r"(?:(?:嗯+|唔+|哦+|噢+|啊+|对+|是的|好(?:的|呀)?|行|可以|了解|明白了?|"
    r"知道了|谢谢(?:你)?|不用谢|你好(?:呀)?|您好|嗨|早上好|下午好|晚上好|晚安|再见|"
    r"ok(?:ay)?|yes|yeah|right|sure|hello|hi|thanks|thankyou|bye))+"
)
_REPLAY_ONLY = re.compile(
    r"(?:(?:请|麻烦(?:你)?|你|刚才|不好意思|我)?"
    r"(?:没(?:有)?听(?:清(?:楚)?|见|到)|声音太小(?:了)?|"
    r"再(?:说|讲|读)(?:一遍|一次)|重复(?:一遍|一次)?|重播)"
    r"(?:一下|吧|好吗|可以吗|了)?)+"
)
_NEW_TOPIC = re.compile(r"^(?:换个话题|换一个话题|另外一个问题|另外问|顺便问|聊点别的)")
_UNSATISFIED = re.compile(
    r"(?:还是|仍然|依然)(?:没(?:有)?(?:懂|明白|理解)|不(?:懂|明白|理解))|"
    r"我(?:没(?:有)?(?:懂|明白|理解)|不(?:懂|明白|理解))|"
    r"(?:没(?:有)?|不够)(?:说清楚|讲清楚|解释清楚|回答清楚)|"
    r"(?:没(?:有)?说清|没(?:有)?讲清|答非所问|不是我问的|不是这个意思|"
    r"没有回答我的问题|没有解决我的问题|不够具体|不够深入|太笼统|讲得太浅|"
    r"回答太简单|我反复问|我已经问|别再重复|还是不清楚|还不清楚)|"
    r"(?:你|刚才|之前|前面|上面).{0,12}(?:说错|答错|不对|没回答|没解释)"
)
_DEEPER_REFERENCE = re.compile(
    r"^(?:(?:那|请|你|能不能|可以|能|麻烦|帮我))*再(?:详细|深入|具体|仔细)|"
    r"(?:刚才|前面|上面|这个问题|这件事|这一点).{0,12}(?:深入|详细|具体|展开)|"
    r"^(?:展开讲讲|展开说说|详细说说|具体说说|继续深入)|"
    r"^(?:(?:能不能|可以|请|你|麻烦|帮我))*(?:详细|深入|具体)(?:一点|一些)"
)
_FOLLOWUP_REFERENCE = re.compile(r"^(?:那|那么|所以|但是|可是|还有)|刚才|前面|这个|这种|这些|这件")
_STRONG_REFERENCE = re.compile(r"刚才|前面|这个|这种|这些|这件")
_FOLLOWUP_ONLY = re.compile(
    r"(?:(?:那|那么|所以|可是|但是))*"
    r"(?:为什么(?:呢|会这样|这么说)?|怎么(?:会这样|回事|解决|办)|"
    r"还有(?:呢|哪些(?:原因|办法|方法|风险|问题))?|"
    r"具体(?:呢|怎么做|怎么实施)|然后呢|接下来呢|能举个例子吗|举个例子|"
    r"这(?:是)?什么原因|这个(?:为什么|怎么解释)|那是什么意思)"
)

# Remove question scaffolding before looking for topic overlap. Sharing words
# such as "为什么", "是多少" or "how" must never establish a common topic.
_SCAFFOLDING = re.compile(
    r"我(?:再问|想问|想了解)(?:一下)?|请(?:你)?帮我|帮我|请问|麻烦(?:你)?|"
    r"告诉我|解释(?:一下)?|分析(?:一下)?|介绍(?:一下)?|请(?:你)?|"
    r"能不能|可不可以|可以|一下|为什么|为啥|如何|怎样|怎么(?:样|办|做)?|"
    r"是什么|什么|是多少|多少|是否|能否|有没有|有何|原因|办法|方法|"
    r"(?:^那|^那么|^所以|^但是|^可是|^还有)|"
    r"这个|那个|这种|这些|这件事|刚才|前面|上面"
)
_ENGLISH_SCAFFOLDING = re.compile(
    r"\b(?:why|how|what|which|when|where|is|are|was|were|do|does|did|can|could|"
    r"would|you|please|tell|me|about|the|a|an|of|to|and|it|this|that|again)\b"
)
_PARTICLES = re.compile(r"[的了呢吗啊吧呀我你他她它及与和]")
_NUMBER = re.compile(r"[0-9零〇一二三四五六七八九十百千万亿两]+")
_TEMPLATE_ONLY = re.compile(
    r"(?:第?[0-9零〇一二三四五六七八九十百千万亿两]+(?:轮|次|个)?(?:提问|问题)|"
    r"(?:提问|问题)[0-9零〇一二三四五六七八九十百千万亿两]+|测试(?:提问|问题))"
)


def _normalize(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum()
    )


def _content(text: str) -> str:
    clean = unicodedata.normalize("NFKC", text).casefold()
    clean = _ENGLISH_SCAFFOLDING.sub(" ", clean)
    clean = _SCAFFOLDING.sub("", clean)
    return _PARTICLES.sub("", _normalize(clean))


def _substantive(text: str) -> bool:
    clean = _normalize(text)
    return bool(
        clean
        and not _HARMLESS.fullmatch(clean)
        and not _REPLAY_ONLY.fullmatch(clean)
        and not _TEMPLATE_ONLY.fullmatch(_content(text))
        and len(_content(text)) >= 2
    )


def _bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _only_values_changed(left: str, right: str) -> bool:
    """A new order number or arithmetic operand is a new request, not a retry."""
    a, b = _content(left), _content(right)
    return (
        a != b
        and bool(_NUMBER.search(a))
        and bool(_NUMBER.search(b))
        and _NUMBER.sub("#", a) == _NUMBER.sub("#", b)
    )


def _same_question(left: str, right: str) -> bool:
    if _only_values_changed(left, right):
        return False
    a, b = _content(left), _content(right)
    if min(len(a), len(b)) < 2:
        return False
    if a == b:
        return True
    shared = _bigrams(a) & _bigrams(b)
    return len(shared) >= 3 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.88


def _same_topic(left: str, right: str, *, strong_reference: bool = False) -> bool:
    if _only_values_changed(left, right):
        return False
    a, b = _bigrams(_content(left)), _bigrams(_content(right))
    if not a or not b:
        return False
    common = len(a & b)
    if common >= 2 and common / min(len(a), len(b)) >= 0.4:
        return True
    # A lone shared two-character topic is weak evidence; require an explicit
    # reference such as "这个" rather than just a generic question prefix.
    return strong_reference and common >= 1 and common / min(len(a), len(b)) >= 0.5


def assess_followup(text: str, history: list[tuple[str, str]]) -> FollowupDecision:
    """Return only locally supported upgrades, with bounded related public Q&A.

    Reasons are stable observability codes. ``none`` means no local evidence,
    not that a fast model is necessarily capable of answering the question.
    """
    clean = _normalize(text)
    if (
        not clean
        or _HARMLESS.fullmatch(clean)
        or _REPLAY_ONLY.fullmatch(clean)
        or _TEMPLATE_ONLY.fullmatch(_content(text))
        or _NEW_TOPIC.search(clean)
    ):
        return FollowupDecision(False, "none", [])
    recent = [
        (question, answer)
        for question, answer in history[-6:]
        if question.strip() and answer.strip() and _substantive(question)
    ]
    if not recent:
        return FollowupDecision(False, "none", [])

    if _UNSATISFIED.search(clean) or _DEEPER_REFERENCE.search(clean):
        related = [pair for pair in recent if _same_topic(text, pair[0])]
        return FollowupDecision(True, "dissatisfied_followup", related or recent[-3:])

    repeated = [pair for pair in recent if _same_question(text, pair[0])]
    if _substantive(text) and repeated:
        return FollowupDecision(True, "repeated_question", repeated)

    if _FOLLOWUP_ONLY.fullmatch(clean):
        return FollowupDecision(True, "topic_followup", recent[-3:])
    strong = bool(_STRONG_REFERENCE.search(clean))
    related = [pair for pair in recent if _same_topic(text, pair[0], strong_reference=strong)]
    # Explicit continuation can refer back to any recent topic. Without it,
    # require overlap with the immediately preceding substantive question.
    if related and (_FOLLOWUP_REFERENCE.search(clean) or _same_topic(text, recent[-1][0])):
        return FollowupDecision(True, "topic_followup", related)
    return FollowupDecision(False, "none", [])
