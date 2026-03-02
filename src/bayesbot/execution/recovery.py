"""Crash recovery — save/restore full system state."""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class CrashRecovery:
    """Saves full system state every 5 minutes.

    On restart:
    1. Load last checkpoint
    2. Reconcile positions with broker
    3. Replay missed bars from DB if needed
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_save: float = 0.0

    def save(self, state: dict) -> None:
        path = self.checkpoint_dir / "latest.json"
        state["_saved_at"] = time.time()
        path.write_text(json.dumps(state, default=str, indent=2))
        self._last_save = time.time()
        logger.debug("Checkpoint saved to {}", path)

    def load(self) -> dict | None:
        path = self.checkpoint_dir / "latest.json"
        if not path.exists():
            return None
        state = json.loads(path.read_text())
        age = time.time() - state.get("_saved_at", 0)
        logger.info("Loaded checkpoint (age={:.0f}s)", age)
        if age > 3600:
            logger.warning("Checkpoint is stale (>1 hour) — state reconstruction recommended")
        return state

    def should_save(self, interval: float = 300.0) -> bool:
        return time.time() - self._last_save > interval
