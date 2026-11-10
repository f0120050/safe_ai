"""T2 execution_only with T2ExecutionCheckV2; otherwise identical to v1 worker."""
import safeai_experiments.runners.t2_execution_worker as worker
from safeai_guardrails.checks.execution_checks_v2 import T2ExecutionCheckV2

worker.T2ExecutionCheck = T2ExecutionCheckV2

if __name__ == "__main__":
    worker.main()
