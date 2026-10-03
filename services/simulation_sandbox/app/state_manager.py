import os
from clickhouse_driver import Client
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass

@dataclass
class WIPState:
    """Snapshot of current Work-in-Progress state"""
    active_cases: pd.DataFrame
    queue_depths: Dict[str, int]
    resource_availability: Dict[str, int]
    timestamp: datetime

class StateManager:
    """Manages Work-in-Progress state from ClickHouse for simulation"""
    
    def __init__(self):
        self.ch_host = os.getenv("CLICKHOUSE_HOST", "localhost")
        self.client = None
    
    def connect(self):
        self.client = Client(host=self.ch_host)
    
    def get_current_state(self) -> WIPState:
        """Get complete WIP snapshot for simulation"""
        if not self.client:
            self.connect()
        
        active_cases = self._get_active_cases()
        queue_depths = self._get_queue_depths()
        resource_availability = self._get_resource_availability()
        
        return WIPState(
            active_cases=active_cases,
            queue_depths=queue_depths,
            resource_availability=resource_availability,
            timestamp=datetime.now()
        )
    
    def _get_active_cases(self) -> pd.DataFrame:
        """Get all cases still in progress"""
        query = """
        SELECT 
            case_id,
            activity,
            status,
            timestamp,
            sla_deadline,
            priority,
            resource
        FROM events
        WHERE case_id NOT IN (
            SELECT case_id FROM events 
            WHERE activity IN ('Approve', 'Reject') AND status = 'COMPLETED'
        )
        ORDER BY case_id, timestamp
        """
        result = self.client.execute(query)
        return pd.DataFrame(result, columns=['case_id', 'activity', 'status', 'timestamp', 'sla_deadline', 'priority', 'resource'])
    
    def _get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depth per activity"""
        query = """
        SELECT activity, count(DISTINCT case_id) as depth
        FROM events
        WHERE status = 'ASSIGNED'
        GROUP BY activity
        """
        result = self.client.execute(query)
        return {row[0]: row[1] for row in result}
    
    def _get_resource_availability(self) -> Dict[str, int]:
        """Get available capacity per resource pool"""
        # Default capacities (from process config)
        default_capacity = {
            "Intake": 5,
            "Credit Team": 8,
            "Review Team": 10,
            "QA Team": 4,
            "Approval Team": 6
        }
        
        # Get currently busy resources
        query = """
        SELECT resource, count(*) as busy
        FROM events
        WHERE status = 'ASSIGNED'
        GROUP BY resource
        """
        result = self.client.execute(query)
        busy = {row[0]: row[1] for row in result}
        
        # Calculate availability
        availability = {}
        for pool, capacity in default_capacity.items():
            pool_busy = sum(v for k, v in busy.items() if pool.lower() in k.lower())
            availability[pool] = max(0, capacity - pool_busy)
        
        return availability
