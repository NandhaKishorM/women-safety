"""
Alert trigger logic for the women safety chip.

The user trigger condition is "a repeated pattern of leg kicks". The classifier
emits one label per 2 s window. The trigger engine fires when:

  1. The current window is `repeated_kick`, or
  2. At least 2 of the last 3 windows are in {hard_kick, repeated_kick}.

The second rule lets the chip catch a sequence of single hard kicks even if
each individual window contains only one kick. A 6 second history (3 windows)
keeps the false positive rate low while still firing within a couple of
seconds of the start of an attack.
"""

from collections import deque

KICK_LABELS = {"hard_kick", "repeated_kick"}


class TriggerEngine:
    def __init__(self, window: int = 3, k: int = 2):
        self.window = window
        self.k = k
        self.history = deque(maxlen=window)

    def step(self, label: str) -> bool:
        self.history.append(label)
        if label == "repeated_kick":
            return True
        kicks = sum(1 for x in self.history if x in KICK_LABELS)
        return kicks >= self.k

    def reset(self):
        self.history.clear()
