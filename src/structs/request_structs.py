from json import JSONEncoder
from typing import List


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class ChangeFormats:
    action: str
    change: str
    reason: str
    position: int

    def __init__(self, action: str, change: str, reason: str, position: int):
        self.action = action
        self.change = change
        self.reason = reason
        self.position = position

    def __str__(self):
        return "{" + f"\"action\":\"{self.action}\", \"change\":\"{self.change}\", \"reason\":\"{self.reason}\", \"position\":\"{self.position}" + "}"


class CheckerRequestFormat:
    filled: bool
    text: str
    list_of_changes: List[ChangeFormats]

    def __init__(self, text):
        self.filled = False
        self.text = text
        self.list_of_changes = []
