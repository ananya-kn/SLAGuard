from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

class BottleneckDetector:
    """Detects bottlenecks in the process flow"""
    
    CAPACITY_THRESHOLDS = {
        "Submit Application": 10,
        "Check Credit": 8,
        "Manual Review": 6,
        "Quality Assurance": 4,
        "Approve": 8,
        "Reject": 8
    }
    
    def detect_bottlenecks(self, queue_depths: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify activities that are becoming bottlenecks"""
        bottlenecks = []
        
        for activity, depth in queue_depths.items():
            threshold = self.CAPACITY_THRESHOLDS.get(activity, 5)
            utilization = depth / threshold
            
            if utilization > 0.8:
                severity = "critical" if utilization > 1.0 else "warning"
                bottlenecks.append({
                    "activity": activity,
                    "queue_depth": depth,
                    "capacity": threshold,
                    "utilization": round(utilization, 2),
                    "severity": severity,
                    "recommendation": self._get_recommendation(activity, utilization)
                })
        
        return sorted(bottlenecks, key=lambda x: -x['utilization'])
    
    def predict_bottleneck_formation(self, queue_depths: Dict[str, int], 
                                      arrival_rate: Dict[str, float]) -> List[Dict[str, Any]]:
        """Predict when bottlenecks will form based on arrival rates"""
        predictions = []
        
        for activity, depth in queue_depths.items():
            threshold = self.CAPACITY_THRESHOLDS.get(activity, 5)
            rate = arrival_rate.get(activity, 1.0)  # cases per hour
            
            if depth < threshold and rate > 0:
                hours_to_bottleneck = (threshold - depth) / rate
                if hours_to_bottleneck < 8:  # Within 8 hours
                    predictions.append({
                        "activity": activity,
                        "current_depth": depth,
                        "hours_to_bottleneck": round(hours_to_bottleneck, 1),
                        "arrival_rate": rate
                    })
        
        return sorted(predictions, key=lambda x: x['hours_to_bottleneck'])
    
    def _get_recommendation(self, activity: str, utilization: float) -> str:
        if utilization > 1.2:
            return f"Critical: Add 2+ resources to {activity} immediately"
        elif utilization > 1.0:
            return f"Add 1 resource to {activity} to clear backlog"
        else:
            return f"Monitor {activity} closely - approaching capacity"
