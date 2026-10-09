"""
Synthetic Data Generator for Appian Predictive Simulation
Generates realistic event logs mimicking Appian's c-column format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
NUM_CASES = 1000
START_DATE = datetime(2024, 1, 1)
PROCESS_ACTIVITIES = [
    "Submit Application",
    "Check Credit",
    "Manual Review",
    "Quality Assurance",
    "Approve",
    "Reject"
]

# Activity durations (hours) - Log-Normal distribution params
ACTIVITY_DURATIONS = {
    "Submit Application": (0.1, 0.05),      # mean, std
    "Check Credit": (2.0, 1.0),
    "Manual Review": (4.0, 2.5),
    "Quality Assurance": (1.5, 0.8),
    "Approve": (0.5, 0.2),
    "Reject": (0.3, 0.1)
}

# Resources
RESOURCES = [f"Agent_{i:03d}" for i in range(1, 21)]

def generate_cases_table(num_cases):
    """Generate static case attributes (Appian Record data)"""
    np.random.seed(42)
    
    cases = pd.DataFrame({
        'case_id': [f"CASE_{i:06d}" for i in range(1, num_cases + 1)],
        'customer_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], num_cases, p=[0.4, 0.3, 0.2, 0.1]),
        'loan_amount': np.random.lognormal(10, 1, num_cases).astype(int),
        'region': np.random.choice(['North', 'South', 'East', 'West'], num_cases),
        'complexity_score': np.random.uniform(1, 10, num_cases).round(2),
        'created_date': [START_DATE + timedelta(hours=np.random.exponential(2)) for _ in range(num_cases)]
    })
    
    return cases

def generate_event_log(cases_df):
    """Generate event log with temporal dynamics and concept drift"""
    events = []
    event_id = 1
    
    for idx, case in cases_df.iterrows():
        case_id = case['case_id']
        current_time = case['created_date']
        complexity = case['complexity_score']
        
        # Process path based on complexity and tier
        if case['customer_tier'] == 'Platinum' or complexity < 3:
            # Fast track: Skip manual review
            activities = ["Submit Application", "Check Credit", "Approve"]
            sla_hours = 24
        elif complexity > 7:
            # Complex cases: Additional QA
            activities = ["Submit Application", "Check Credit", "Manual Review", "Quality Assurance"]
            # 30% rejection rate for complex cases
            if random.random() < 0.3:
                activities.append("Reject")
            else:
                activities.append("Approve")
            sla_hours = 72
        else:
            # Standard path
            activities = ["Submit Application", "Check Credit", "Manual Review", "Approve"]
            sla_hours = 48
        
        # Calculate SLA deadline
        sla_deadline = case['created_date'] + timedelta(hours=sla_hours)
        
        # Generate events for each activity
        for activity in activities:
            resource = random.choice(RESOURCES)
            
            # Duration with complexity modifier
            base_mean, base_std = ACTIVITY_DURATIONS[activity]
            duration_modifier = 1 + (complexity - 5) * 0.1  # Complexity affects duration
            duration = max(0.1, np.random.lognormal(np.log(base_mean * duration_modifier), base_std))
            
            # Inject concept drift (Month 4+: 50% slower for Manual Review)
            if activity == "Manual Review" and current_time > START_DATE + timedelta(days=90):
                duration *= 1.5
            
            # ASSIGNED status
            events.append({
                'event_id': event_id,
                'case_id': case_id,
                'activity': activity,
                'status': 'ASSIGNED',
                'timestamp': current_time,
                'resource': resource,
                'sla_deadline': sla_deadline,
                'priority': 1 if case['customer_tier'] == 'Platinum' else 3
            })
            event_id += 1
            
            # COMPLETED status
            completion_time = current_time + timedelta(hours=duration)
            events.append({
                'event_id': event_id,
                'case_id': case_id,
                'activity': activity,
                'status': 'COMPLETED',
                'timestamp': completion_time,
                'resource': resource,
                'sla_deadline': sla_deadline,
                'priority': 1 if case['customer_tier'] == 'Platinum' else 3
            })
            event_id += 1
            
            current_time = completion_time
    
    return pd.DataFrame(events)

def transform_to_appian_ccolumn_format(events_df):
    """Transform to Appian's c-column format for API compatibility"""
    appian_format = events_df.copy()
    appian_format.columns = [f'c{i}' for i in range(len(appian_format.columns))]
    
    # Standard mapping (as per PDF):
    # c0: Activity, c1: Status, c2: Timestamp, c3: SLA, c4: Priority, c5: Case ID
    appian_format = appian_format[[1, 3, 4, 6, 7, 2]]  # Reorder columns
    appian_format.columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    
    return appian_format

def calculate_remaining_time(events_df):
    """Calculate remaining time to SLA for each event (ML target variable)"""
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

# Generate the data
print("Generating cases...")
cases = generate_cases_table(NUM_CASES)

print("Generating event log...")
events = generate_event_log(cases)

print("Calculating ML targets...")
events = calculate_remaining_time(events)

# Save outputs
cases.to_csv('cases_table.csv', index=False)
events.to_csv('events_log.csv', index=False)

# Also save in Appian c-column format
appian_format = transform_to_appian_ccolumn_format(events)
appian_format.to_csv('appian_ccolumn_format.csv', index=False)

print(f"\n✅ Generated {len(cases)} cases with {len(events)} events")
print(f"\nCases Table Shape: {cases.shape}")
print(f"Events Table Shape: {events.shape}")
print(f"\nSample Case:\n{cases.head(3)}")
print(f"\nSample Events:\n{events.head(6)}")
print(f"\nSLA Breach Rate: {events[events['status']=='COMPLETED'].groupby('case_id')['will_breach_sla'].first().mean():.2%}")