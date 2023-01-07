from json import JSONEncoder
from typing import List


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class ChangeFormats:
    before: str
    after: str
    reason: str

    def __init__(self, before: str, after: str, reason: str):
        self.before = before
        self.after = after
        self.reason = reason


class CheckerRequestFormat:
    filled: bool
    text: str
    list_of_changes: List[ChangeFormats]

    def __init__(self, text):
        self.filled = False
        self.text = text
        self.list_of_changes = []
