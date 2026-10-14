import os
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

class SLAForecaster:
    """Predicts future SLA breaches using inference service"""
    
    def __init__(self):
        self.inference_url = os.getenv("INFERENCE_URL", "http://localhost:8004")
    
    async def forecast_breaches(self, active_cases: pd.DataFrame, horizon_hours: int = 24) -> List[Dict[str, Any]]:
        """Predict which cases will breach SLA within horizon"""
        breaches = []
        now = datetime.now()
        
        for _, case in active_cases.groupby('case_id').last().iterrows():
            # Calculate elapsed time
            case_start = active_cases[active_cases['case_id'] == case.name]['timestamp'].min()
            elapsed_hours = (now - case_start).total_seconds() / 3600
            
            # Time until SLA deadline
            time_to_sla = (case['sla_deadline'] - now).total_seconds() / 3600
            
            # Simple rule-based prediction (fallback if ML not available)
            # Will be replaced by ML once models are trained
            breach_probability = self._calculate_breach_probability(
                elapsed_hours=elapsed_hours,
                time_to_sla=time_to_sla,
                current_activity=case['activity'],
                priority=case['priority']
            )
            
            if breach_probability > 0.5 and time_to_sla < horizon_hours:
                breaches.append({
                    "case_id": case.name,
                    "current_activity": case['activity'],
                    "breach_probability": breach_probability,
                    "hours_to_deadline": time_to_sla,
                    "priority": case['priority']
                })
        
        # Sort by probability (highest risk first)
        return sorted(breaches, key=lambda x: -x['breach_probability'])
    
    def _calculate_breach_probability(self, elapsed_hours: float, time_to_sla: float, 
                                       current_activity: str, priority: int) -> float:
        """Simple rule-based breach probability (placeholder for ML)"""
        # Estimate remaining work based on current activity
        remaining_estimates = {
            "Submit Application": 8.0,
            "Check Credit": 6.0,
            "Manual Review": 4.0,
            "Quality Assurance": 2.0,
            "Approve": 0.5,
            "Reject": 0.5
        }
        estimated_remaining = remaining_estimates.get(current_activity, 4.0)
        
        # Calculate probability based on time buffer
        buffer = time_to_sla - estimated_remaining
        if buffer < 0:
            return 0.95  # Very likely to breach
        elif buffer < 2:
            return 0.7
        elif buffer < 4:
            return 0.4
        else:
            return 0.1
