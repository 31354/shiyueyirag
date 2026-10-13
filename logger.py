# logger.py
import json
import time
from pathlib import Path

LOG_DIR = Path("./agent_logs")
LOG_DIR.mkdir(exist_ok=True)


def _ts():
    return time.strftime("%Y%m%d_%H%M%S")


class AgentLogger:
    def __init__(self, session_name=None):
        self.session_name = session_name or _ts()
        self.session_file = LOG_DIR / f"session_{self.session_name}.jsonl"
        self.turn = 0
        self.tool_call_count = 0

    def _write(self, record: dict):
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_user(self, text: str):
        self._write({"type": "user", "content": text})

    def log_turn_start(self, turn: int, tool_choice):
        self.turn = turn
        self._write({"type": "turn_start", "turn": turn, "tool_choice": tool_choice})

    def log_tool_call(self, name: str, args: str):
        self.tool_call_count += 1
        self._write({
            "type": "tool_call",
            "turn": self.turn,
            "name": name,
            "args": args,
        })

    def log_tool_result(self, name: str, result: str, elapsed: float):
        self._write({
            "type": "tool_result",
            "turn": self.turn,
            "name": name,
            "result": result,
            "elapsed_sec": round(elapsed, 3),
        })

    def log_answer(self, content: str, elapsed: float):
        self._write({
            "type": "answer",
            "content": content,
            "elapsed_sec": round(elapsed, 3),
            "total_tool_calls": self.tool_call_count,
        })

    def log_error(self, where: str, error: str):
        self._write({"type": "error", "where": where, "error": error})