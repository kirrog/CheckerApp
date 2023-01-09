import difflib
from typing import List

from src.structs.request_structs import ChangeFormats


def find_changes(before: str, after: str) -> List[ChangeFormats]:
    r = []
    pos_changes = 0
    neg_changes = 0
    num = 0
    prev = " "
    collector_string = ""
    i = 0
    for i, s in enumerate(difflib.ndiff(before, after)):
        if s[0] == ' ':
            if prev != " ":
                if prev == "-":
                    chf = ChangeFormats("del", collector_string, "punctuation", i - pos_changes - num)
                else:
                    chf = ChangeFormats("add", collector_string, "punctuation", i - neg_changes - num)
                r.append(chf)
                collector_string = ""
                num = 0
            prev = " "
        elif s[0] == '-':
            if prev == '-':
                collector_string += s[-1]
                num += 1
            elif prev == "+":
                chf = ChangeFormats("add", collector_string, "punctuation", i - neg_changes - num)
                r.append(chf)
                collector_string = s[-1]
                num = 1
            else:
                collector_string += s[-1]
                num += 1
            neg_changes += 1
            prev = "-"
        elif s[0] == '+':
            if prev == '+':
                collector_string += s[-1]
                num += 1
            elif prev == "-":
                chf = ChangeFormats("add", collector_string, "punctuation", i - pos_changes - num + 1)
                r.append(chf)
                collector_string = s[-1]
                num = 1
            else:
                collector_string += s[-1]
                num += 1
            pos_changes += 1
            prev = "+"
    if prev != " ":
        if prev == "-":
            chf = ChangeFormats("del", collector_string, "punctuation", i - pos_changes - num)
        else:
            chf = ChangeFormats("add", collector_string, "punctuation", i - neg_changes - num)
        r.append(chf)
    return r