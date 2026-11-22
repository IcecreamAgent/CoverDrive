"""Polite spoken address from verified customer fields, never name-based gender."""

import re

_COMPOUND_SURNAMES = frozenset("欧阳 太史 端木 上官 司马 东方 独孤 南宫 万俟 闻人 夏侯 诸葛 尉迟 公羊 赫连 澹台 皇甫 宗政 濮阳 公冶 太叔 申屠 公孙 慕容 仲孙 钟离 长孙 宇文 司徒 鲜于 司空 闾丘 子车 亓官 司寇 巫马 公西 颛孙 壤驷 公良 漆雕 乐正 宰父 谷梁 拓跋 夹谷 轩辕 令狐 段干 百里 呼延 东郭 南门 羊舌 微生 梁丘 左丘 东门 西门 第五".split())


def customer_salutation(customer: dict) -> str:
    """Explicit address wins; without known gender or surname, say 您."""
    for key in ("称呼", "称谓", "salutation"):
        value = customer.get(key)
        if isinstance(value, str) and re.fullmatch(r"[\u3400-\u9fff]{1,4}(?:先生|女士)", value.strip()):
            return value.strip()
    genders = [customer[key].strip().lower() for key in ("性别", "gender", "sex")
               if isinstance(customer.get(key), str)]
    titles = {"男": "先生", "男性": "先生", "male": "先生", "m": "先生",
              "女": "女士", "女性": "女士", "female": "女士", "f": "女士"}
    known_titles = {titles[value] for value in genders if value in titles}
    if len(known_titles) != 1:
        return "您"
    surname = customer.get("姓氏") or customer.get("surname")
    if not isinstance(surname, str) or not re.fullmatch(r"[\u3400-\u9fff]{1,4}", surname.strip()):
        name = customer.get("姓名")
        if not isinstance(name, str) or not re.fullmatch(r"[\u3400-\u9fff]{2,4}", name.strip()):
            return "您"
        name = name.strip()
        surname = name[:2] if name[:2] in _COMPOUND_SURNAMES else name[:1]
    return surname.strip() + next(iter(known_titles))
