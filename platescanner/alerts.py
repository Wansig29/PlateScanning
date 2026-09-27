"""What the identity dashboard shows when violators arrive.

Kept free of Qt so the rules can be tested directly. The guiding rule: a
violation alert never disappears until a guard has acknowledged it.

- A violation stays on the dashboard until acknowledged; nothing replaces
  it in the meantime, however long nobody is watching.
- Violators arriving meanwhile queue up (oldest first) and each
  Acknowledge brings up the next.
- Looking at an older scan from the Logs doesn't lose anything: the
  violator on screen goes back to the front of the queue, and the dashboard
  offers a "Back to violations" button.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class DashboardQueue:
    def __init__(self):
        self.current: Any = None      # unacknowledged violation on screen
        self.waiting: deque = deque()  # violations queued behind it
        self.viewing = False           # the guard opened another scan from the Logs

    def pending(self) -> int:
        """Violations not acknowledged yet (on screen or queued)."""
        return len(self.waiting) + (self.current is not None)

    def locked(self) -> bool:
        """True while the dashboard must not be replaced by a newly scanned vehicle."""
        return self.viewing or self.current is not None

    def on_violation(self, item: Any) -> Any:
        """Returns the item to show now, or None if it was queued."""
        if self.locked():
            self.waiting.append(item)
            return None
        self.current = item
        return item

    def on_clear(self) -> bool:
        """A non-violating vehicle was scanned: may it replace what's on screen?"""
        return not self.locked()

    def acknowledge(self) -> tuple[Any, Any]:
        """Acknowledge pressed. Returns (the item acknowledged, the next one to show)."""
        done, self.current = self.current, None
        self.viewing = False
        if self.waiting:
            self.current = self.waiting.popleft()
        return done, self.current

    def back(self) -> Any:
        """"Back to violations" pressed while viewing an older scan. Returns the item to show."""
        self.viewing = False
        if self.current is None and self.waiting:
            self.current = self.waiting.popleft()
        return self.current

    def view_other(self) -> bool:
        """The guard opened another scan. Returns True if violations are waiting
        (the dashboard should then offer a way back to them)."""
        if self.current is not None:
            self.waiting.appendleft(self.current)
            self.current = None
        self.viewing = bool(self.waiting)
        return self.viewing
