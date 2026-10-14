import os
from clickhouse_driver import Client
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd

class WIPMonitor:
    """Monitors Work-in-Progress state from ClickHouse"""
    
    def __init__(self):
        self.ch_host = os.getenv("CLICKHOUSE_HOST", "localhost")
        self.client = None
    
    def connect(self):
        self.client = Client(host=self.ch_host)
    
    def get_active_cases(self) -> pd.DataFrame:
        """Get all cases that are still in progress (no terminal activity)"""
        query = """
        SELECT 
            case_id,
            activity,
            status,
            timestamp,
            sla_deadline,
            priority
        FROM events
        WHERE case_id NOT IN (
            SELECT case_id FROM events 
            WHERE activity IN ('Approve', 'Reject') AND status = 'COMPLETED'
        )
        ORDER BY case_id, timestamp
        """
        if not self.client:
            self.connect()
        result = self.client.execute(query)
        return pd.DataFrame(result, columns=['case_id', 'activity', 'status', 'timestamp', 'sla_deadline', 'priority'])
    
    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depth per activity"""
        query = """
        SELECT activity, count(*) as queue_depth
        FROM events
        WHERE status = 'ASSIGNED'
        AND event_id NOT IN (
            SELECT e2.event_id FROM events e1
            JOIN events e2 ON e1.case_id = e2.case_id AND e1.activity = e2.activity
            WHERE e1.status = 'COMPLETED' AND e2.status = 'ASSIGNED'
        )
        GROUP BY activity
        """
        if not self.client:
            self.connect()
        result = self.client.execute(query)
        return {row[0]: row[1] for row in result}
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization per resource pool"""
        query = """
        SELECT resource, count(*) as active_tasks
        FROM events
        WHERE status = 'ASSIGNED'
        GROUP BY resource
        """
        if not self.client:
            self.connect()
        result = self.client.execute(query)
        # Assume 8 hours max per resource for utilization calc
        return {row[0]: min(1.0, row[1] / 8.0) for row in result}
