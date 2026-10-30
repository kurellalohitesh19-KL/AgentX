"""SpendWise AI - Tool Registry.

Defines the base Tool interface and registers both core system tools and
specialized financial analytics tools.
"""


class Tool:
    name = "tool"
    description = ""
    args_doc = "{}"

    def execute(self, **kwargs):
        raise NotImplementedError


def build_tools():
    # Core system & query tools
    from tools.calculator import Calculator
    from tools.data_query import ListFiles, ReadFile, SqlQuery
    from tools.web_search import WebSearch

    # SpendWise AI financial analytics tools
    from tools.categorizer import Categorizer
    from tools.duplicate_detector import DuplicateDetector
    from tools.anomaly_detector import AnomalyDetector
    from tools.subscription_detector import SubscriptionDetector
    from tools.budget_planner import BudgetPlanner
    from tools.risk_scorer import RiskScorerTool

    tools = [
        ListFiles(),
        ReadFile(),
        SqlQuery(),
        Calculator(),
        WebSearch(),
        Categorizer(),
        DuplicateDetector(),
        AnomalyDetector(),
        SubscriptionDetector(),
        BudgetPlanner(),
        RiskScorerTool(),
    ]
    return {t.name: t for t in tools}
