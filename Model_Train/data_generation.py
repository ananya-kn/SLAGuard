"""
Unified Data Generator for Model Training
Handles all 4 variability scenarios with format matching the inference pipeline

This is the SINGLE SOURCE OF TRUTH for data generation.
Matches format with: services/data_simulator/app/generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional
from tqdm import tqdm

# Configuration
NUM_CASES = 10000
START_DATE = datetime(2024, 1, 1)

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

# Shift availability by hour (for workforce variability)
SHIFT_AVAILABILITY = {
    0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.3,
    6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 0.8, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.9, 17: 0.7,
    18: 0.5, 19: 0.4, 20: 0.3, 21: 0.3, 22: 0.2, 23: 0.2
}


def generate_cases_table(num_cases: int = NUM_CASES, 
                         start_date: datetime = START_DATE,
                         use_realistic_arrivals: bool = True) -> pd.DataFrame:
    """
    Generate cases with all variability factors:
    - Non-homogeneous Poisson arrivals (volume spikes)
    - Complexity distribution
    - Customer tier distribution
    """
    np.random.seed(42)
    
    if use_realistic_arrivals:
        arrival_times = _generate_realistic_arrivals(num_cases, start_date)
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


def _generate_realistic_arrivals(num_cases: int, start_date: datetime) -> List[datetime]:
    """Non-homogeneous Poisson: Peak during business hours, low at night/weekend"""
    arrivals = []
    current_time = start_date
    
    # Disabled progress bar to avoid memory issues in Colab
    for _ in tqdm(range(num_cases), desc="Generating Cases", disable=True):
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Base rate: 2 cases/hour
        base_rate = 2.0
        
        # Hour modifier
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


def _setup_workforce_variability(date: datetime, unavailable_pct: float = 0.15) -> List[str]:
    """Mark random agents as unavailable (sick/holiday)"""
    num_unavailable = int(len(RESOURCES) * unavailable_pct)
    return random.sample(RESOURCES, num_unavailable)


def _get_available_resources(timestamp: datetime, unavailable_agents: List[str]) -> List[str]:
    """Get available resources considering shift and unavailability"""
    # Filter out unavailable
    available = [r for r in RESOURCES if r not in unavailable_agents]
    
    # Apply shift availability
    hour = timestamp.hour
    shift_pct = SHIFT_AVAILABILITY.get(hour, 0.5)
    num_on_shift = max(1, int(len(available) * shift_pct))
    
    return available[:num_on_shift]


def generate_event_log(cases_df: pd.DataFrame,
                       apply_concept_drift: bool = True,
                       drift_start_days: int = 90) -> pd.DataFrame:
    """
    Generate event log with all variability factors:
    - Workforce variability (shift schedules, sick agents)
    - Complexity-based duration
    - Concept drift (Month 4+: Manual Review 50% slower)
    
    OUTPUT FORMAT (matches inference pipeline):
    - event_id, case_id, activity, status, timestamp, resource
    - sla_deadline, priority
    - hour_of_day, day_of_week, available_resources_count
    - queue_depth_estimate, complexity_score, duration_hours
    """
    events = []
    event_id = 1
    
    # Setup workforce variability per day
    unavailable_by_day = {}
    if not cases_df.empty:
        min_date = cases_df['created_date'].min()
        max_date = cases_df['created_date'].max()
        current = min_date
        while current <= max_date + timedelta(days=7):
            day_str = current.strftime("%Y-%m-%d")
            unavailable_by_day[day_str] = _setup_workforce_variability(current)
            current += timedelta(days=1)
    
    start_date = cases_df['created_date'].min() if not cases_df.empty else START_DATE
    
    # Disabled progress bar to avoid memory issues in Colab
    for idx, case in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Generating Events", disable=True):
        case_id = case['case_id']
        current_time = case['created_date']
        complexity = case['complexity_score']
        
        # Process path based on complexity and tier
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
            # Get available resources
            day_str = current_time.strftime("%Y-%m-%d")
            unavailable = unavailable_by_day.get(day_str, [])
            available = _get_available_resources(current_time, unavailable)
            resource = random.choice(available) if available else random.choice(RESOURCES)
            available_count = len(available)
            
            # Calculate duration with all modifiers
            base_mean, base_std = ACTIVITY_DURATIONS[activity]
            duration_modifier = 1 + (complexity - 5) * 0.1
            
            # Concept drift: Month 4+ Manual Review 50% slower
            if apply_concept_drift and activity == "Manual Review":
                days_elapsed = (current_time - start_date).days
                if days_elapsed > drift_start_days:
                    duration_modifier *= 1.5
            
            # Workforce shortage modifier
            if available_count < 10:
                duration_modifier *= 1 + (10 - available_count) * 0.05
            
            duration = max(0.1, np.random.lognormal(np.log(base_mean * duration_modifier), base_std))
            
            # Queue depth estimate
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
                # ML features (matching inference pipeline)
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday(),
                'available_resources_count': available_count,
                'queue_depth_estimate': queue_estimate,
                'complexity_score': complexity,
                'customer_tier': case['customer_tier'],
                'region': case['region'],
                'loan_amount': case['loan_amount']
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
                'customer_tier': case['customer_tier'],
                'region': case['region'],
                'loan_amount': case['loan_amount'],
                'duration_hours': duration
            })
            event_id += 1
            current_time = completion_time
    
    return pd.DataFrame(events)


def calculate_remaining_time(events_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate remaining time and SLA breach target"""
    events_df = events_df.sort_values(['case_id', 'timestamp'])
    
    # Get last completion time per case
    case_completion = events_df[events_df['status'] == 'COMPLETED'].groupby('case_id')['timestamp'].max()
    
    # Merge back
    events_df['case_completion_time'] = events_df['case_id'].map(case_completion)
    events_df['remaining_time_hours'] = (
        (events_df['case_completion_time'] - events_df['timestamp']).dt.total_seconds() / 3600
    )
    
    # Binary target: Will breach SLA?
    events_df['will_breach_sla'] = (
        events_df['case_completion_time'] > events_df['sla_deadline']
    ).astype(int)
    
    return events_df


def generate_training_data(num_cases: int = NUM_CASES, 
                           output_prefix: str = "") -> tuple:
    """
    Generate complete training dataset with all variability factors.
    
    Returns:
        (cases_df, events_df)
    """
    print("🏭 Generating Training Data with All Variability Factors...")
    print("=" * 60)
    
    print("📊 Variability factors enabled:")
    print("  1. ✅ Volume Spikes (non-homogeneous Poisson arrivals)")
    print("  2. ✅ Workforce Variability (shift schedules, sick agents)")
    print("  3. ✅ Case Complexity (duration modifiers, process branching)")
    print("  4. ✅ Concept Drift (Month 4+ Manual Review 50% slower)")
    
    print(f"\n📋 Generating {num_cases} cases...")
    cases = generate_cases_table(num_cases)
    
    print("🔄 Generating event log...")
    events = generate_event_log(cases)
    
    print("⏱️ Calculating ML targets...")
    events = calculate_remaining_time(events)
    
    # Save outputs
    cases_file = f"{output_prefix}cases_table.csv" if output_prefix else "cases_table.csv"
    events_file = f"{output_prefix}events_log.csv" if output_prefix else "events_log.csv"
    
    cases.to_csv(cases_file, index=False)
    events.to_csv(events_file, index=False)
    
    print(f"\n✅ Generated {len(cases)} cases with {len(events)} events")
    print(f"📊 SLA Breach Rate: {events[events['status']=='COMPLETED'].groupby('case_id')['will_breach_sla'].first().mean():.2%}")
    print(f"💾 Saved to: {cases_file}, {events_file}")
    
    # Print feature summary
    print("\n📋 Features available for training:")
    feature_cols = ['hour_of_day', 'day_of_week', 'available_resources_count', 
                    'queue_depth_estimate', 'complexity_score', 'priority']
    for col in feature_cols:
        if col in events.columns:
            print(f"  - {col}")
    
    return cases, events


if __name__ == "__main__":
    cases, events = generate_training_data()
    print(f"\n📊 Sample Cases:\n{cases.head(3)}")
    print(f"\n📊 Sample Events:\n{events.head(6)}")
