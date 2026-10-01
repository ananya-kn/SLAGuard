import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional

# Configuration
PROCESS_ACTIVITIES = [
    "Submit Application",
    "Check Credit",
    "Manual Review",
    "Quality Assurance",
    "Approve",
    "Reject"
]

ACTIVITY_DURATIONS = {
    "Submit Application": (0.1, 0.05),
    "Check Credit": (2.0, 1.0),
    "Manual Review": (4.0, 2.5),
    "Quality Assurance": (1.5, 0.8),
    "Approve": (0.5, 0.2),
    "Reject": (0.3, 0.1)
}

RESOURCES = [f"Agent_{i:03d}" for i in range(1, 21)]

# Shift schedules (resource availability by hour)
SHIFT_AVAILABILITY = {
    # Hour: % of resources available
    0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.3,
    6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 0.8, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.5, 19: 0.4, 20: 0.3, 21: 0.3, 22: 0.2, 23: 0.2
}

class DataGenerator:
    """Enhanced data generator with variability handling"""
    
    def __init__(self):
        self.unavailable_agents: Dict[str, List[str]] = {}  # date -> list of unavailable agents
    
    def generate_cases(self, num_cases: int, start_date: datetime, 
                       use_realistic_arrivals: bool = True) -> pd.DataFrame:
        """Generate cases with optional non-homogeneous Poisson arrivals"""
        np.random.seed(42)
        
        if use_realistic_arrivals:
            # Non-homogeneous Poisson: higher rate during business hours
            arrival_times = self._generate_realistic_arrivals(num_cases, start_date)
        else:
            arrival_times = [start_date + timedelta(hours=np.random.exponential(2)) for _ in range(num_cases)]
        
        cases = pd.DataFrame({
            'case_id': [f"CASE_{i:06d}" for i in range(1, num_cases + 1)],
            'customer_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_cases, p=[0.4, 0.3, 0.2, 0.1]),
            'loan_amount': np.random.lognormal(10, 1, num_cases).astype(int),
            'region': np.random.choice(['North', 'South', 'East', 'West'], num_cases),
            'complexity_score': np.random.uniform(1, 10, num_cases).round(2),
            'created_date': arrival_times
        })
        return cases
    
    def _generate_realistic_arrivals(self, num_cases: int, start_date: datetime) -> List[datetime]:
        """Generate arrivals with non-homogeneous Poisson (business hours peak)"""
        arrivals = []
        current_time = start_date
        
        for _ in range(num_cases):
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            # Base rate: 2 cases/hour
            base_rate = 2.0
            
            # Hour modifier (peak 9-17, low at night)
            hour_modifier = {
                0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
                6: 0.3, 7: 0.5, 8: 0.8, 9: 1.2, 10: 1.5, 11: 1.3,
                12: 0.8, 13: 1.4, 14: 1.5, 15: 1.3, 16: 1.0, 17: 0.7,
                18: 0.4, 19: 0.3, 20: 0.2, 21: 0.2, 22: 0.1, 23: 0.1
            }.get(hour, 0.5)
            
            # Weekend modifier
            weekend_modifier = 0.3 if day_of_week >= 5 else 1.0
            
            effective_rate = base_rate * hour_modifier * weekend_modifier
            inter_arrival = np.random.exponential(1.0 / max(0.1, effective_rate))
            
            current_time = current_time + timedelta(hours=inter_arrival)
            arrivals.append(current_time)
        
        return arrivals
    
    def setup_workforce_variability(self, date: datetime, unavailable_pct: float = 0.15):
        """Mark random agents as unavailable (sick/holiday)"""
        date_str = date.strftime("%Y-%m-%d")
        num_unavailable = int(len(RESOURCES) * unavailable_pct)
        self.unavailable_agents[date_str] = random.sample(RESOURCES, num_unavailable)
    
    def get_available_resources(self, timestamp: datetime) -> List[str]:
        """Get available resources at a given time"""
        date_str = timestamp.strftime("%Y-%m-%d")
        unavailable = self.unavailable_agents.get(date_str, [])
        
        # Filter by shift availability
        hour = timestamp.hour
        shift_pct = SHIFT_AVAILABILITY.get(hour, 0.5)
        available = [r for r in RESOURCES if r not in unavailable]
        
        # Further reduce by shift
        num_on_shift = max(1, int(len(available) * shift_pct))
        return available[:num_on_shift]

    def generate_events(self, cases_df: pd.DataFrame, 
                        apply_concept_drift: bool = True,
                        drift_start_days: int = 90) -> List[Dict]:
        """Generate events with all variability factors"""
        events = []
        event_id = 1
        
        # Setup workforce variability
        if not cases_df.empty:
            min_date = cases_df['created_date'].min()
            max_date = cases_df['created_date'].max()
            current = min_date
            while current <= max_date + timedelta(days=7):
                self.setup_workforce_variability(current)
                current += timedelta(days=1)
        
        for idx, case in cases_df.iterrows():
            case_id = case['case_id']
            current_time = case['created_date']
            complexity = case['complexity_score']
            
            # Process path based on complexity
            if case['customer_tier'] == 'Platinum' or complexity < 3:
                activities = ["Submit Application", "Check Credit", "Approve"]
                sla_hours = 24
            elif complexity > 7:
                activities = ["Submit Application", "Check Credit", "Manual Review", "Quality Assurance"]
                activities.append("Reject" if random.random() < 0.3 else "Approve")
                sla_hours = 72
            else:
                activities = ["Submit Application", "Check Credit", "Manual Review", "Approve"]
                sla_hours = 48
            
            sla_deadline = case['created_date'] + timedelta(hours=sla_hours)
            
            for activity in activities:
                # Get available resource at this time
                available = self.get_available_resources(current_time)
                resource = random.choice(available) if available else random.choice(RESOURCES)
                
                # Calculate available_resources_count (for ML feature)
                available_count = len(available)
                
                # Base duration with complexity modifier
                base_mean, base_std = ACTIVITY_DURATIONS[activity]
                duration_modifier = 1 + (complexity - 5) * 0.1
                
                # Concept drift: Month 4+ Manual Review is 50% slower
                if apply_concept_drift and activity == "Manual Review":
                    days_elapsed = (current_time - cases_df['created_date'].min()).days
                    if days_elapsed > drift_start_days:
                        duration_modifier *= 1.5  # 50% slower
                
                # Workforce shortage: fewer resources = longer wait
                resource_shortage_modifier = 1.0
                if available_count < 10:
                    resource_shortage_modifier = 1 + (10 - available_count) * 0.05
                
                duration = max(0.1, np.random.lognormal(
                    np.log(base_mean * duration_modifier * resource_shortage_modifier), 
                    base_std
                ))
                
                # Calculate queue depth at this moment (simplified)
                hour = current_time.hour
                queue_estimate = int(10 * (1 - SHIFT_AVAILABILITY.get(hour, 0.5)))
                
                # ASSIGNED event
                events.append({
                    'event_id': event_id,
                    'case_id': case_id,
                    'activity': activity,
                    'status': 'ASSIGNED',
                    'timestamp': current_time,
                    'resource': resource,
                    'sla_deadline': sla_deadline,
                    'priority': 1 if case['customer_tier'] == 'Platinum' else 3,
                    # ML features
                    'hour_of_day': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'available_resources_count': available_count,
                    'queue_depth_estimate': queue_estimate,
                    'complexity_score': complexity,
                    # c-column aliases
                    'c0': activity,
                    'c1': 'ASSIGNED',
                    'c2': current_time,
                    'c3': sla_deadline,
                    'c4': 1 if case['customer_tier'] == 'Platinum' else 3,
                    'c5': case_id
                })
                event_id += 1
                
                # COMPLETED event
                completion_time = current_time + timedelta(hours=duration)
                events.append({
                    'event_id': event_id,
                    'case_id': case_id,
                    'activity': activity,
                    'status': 'COMPLETED',
                    'timestamp': completion_time,
                    'resource': resource,
                    'sla_deadline': sla_deadline,
                    'priority': 1 if case['customer_tier'] == 'Platinum' else 3,
                    # ML features
                    'hour_of_day': completion_time.hour,
                    'day_of_week': completion_time.weekday(),
                    'available_resources_count': available_count,
                    'queue_depth_estimate': queue_estimate,
                    'complexity_score': complexity,
                    'duration_hours': duration,
                    # c-column aliases
                    'c0': activity,
                    'c1': 'COMPLETED',
                    'c2': completion_time,
                    'c3': sla_deadline,
                    'c4': 1 if case['customer_tier'] == 'Platinum' else 3,
                    'c5': case_id
                })
                event_id += 1
                current_time = completion_time
                
        return events
