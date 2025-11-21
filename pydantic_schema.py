# pydantic_schema.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pydantic as pd  # works in v1 and v2

class RiskFlag(BaseModel):
    name: str
    reason: str
    severity: str  # "low" | "medium" | "high"

class PortfolioSummary(BaseModel):
    dso_days_90d: Optional[float]
    aging_overview: Dict[str, int]  # {"30+": int, "60+": int, "90+": int}
    high_utilization_accounts: List[Dict[str, Any]] = Field(default_factory=list)
    failed_promises_30d: int
    key_risks: List[RiskFlag] = Field(default_factory=list)
    notes: Optional[str] = None

def validate_portfolio_summary(raw_json: str) -> "PortfolioSummary":
    """
    Version-agnostic JSON validation:
    - Pydantic v2: uses model_validate_json
    - Pydantic v1: uses parse_raw
    """
    if hasattr(PortfolioSummary, "model_validate_json"):  # v2
        return PortfolioSummary.model_validate_json(raw_json)
    # v1 fallback
    return PortfolioSummary.parse_raw(raw_json)
