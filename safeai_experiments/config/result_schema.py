from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timezone
import json


@dataclass
class ExperimentResult:
    run_id: str
    pair_id: str
    model: str
    attack_id: str
    safety_mode: str
    repetition: int

    status: str = "PENDING"
    input_decision: Optional[str] = None
    execution_decision: Optional[str] = None
    output_decision: Optional[str] = None

    agent_executed: bool = False
    unauthorized_http_access: Optional[bool] = None
    unauthorized_final_disclosure: Optional[bool] = None

    total_time_s: Optional[float] = None
    input_check_time_s: Optional[float] = None
    output_check_time_s: Optional[float] = None

    # Technical reliability metrics
    json_error_count: int = 0
    tool_call_error_count: int = 0
    timeout_occurred: bool = False
    agent_step_count: Optional[int] = None
    task_completed: Optional[bool] = None

    error_type: Optional[str] = None
    error_message: Optional[str] = None

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(
                timezone.utc
            ).isoformat()

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False
        )
