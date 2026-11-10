"""T4 execution_only with T4ExecutionCheckV2 and JuiceShopAdapterV2."""
import safeai_experiments.runners.t4_worker as worker
from safeai_guardrails.adapters.juice_shop_adapter_v2 import JuiceShopAdapterV2
from safeai_guardrails.checks.execution_checks_v2 import T4ExecutionCheckV2

worker.T4ExecutionCheck = T4ExecutionCheckV2
worker.JuiceShopAdapter = JuiceShopAdapterV2

if __name__ == "__main__":
    worker.main()
