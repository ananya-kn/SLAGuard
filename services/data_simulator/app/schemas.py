from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Event(BaseModel):
    event_id: int
    case_id: str
    activity: str
    status: str
    timestamp: datetime
    resource: str
    sla_deadline: datetime
    priority: int
    
    # Appian c-column aliases (optional, for output)
    c0: Optional[str] = None
    c1: Optional[str] = None
    c2: Optional[datetime] = None
    c3: Optional[datetime] = None
    c4: Optional[int] = None
    c5: Optional[str] = None

class GenerationRequest(BaseModel):
    num_cases: int = 10
    start_date: Optional[datetime] = None
