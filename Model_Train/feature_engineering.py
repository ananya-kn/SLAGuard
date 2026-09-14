"""
Feature Engineering Pipeline for Appian Process Mining Data
Creates features suitable for XGBoost SLA breach prediction
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ProcessFeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        self.target_column = 'will_breach_sla'
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamps"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['is_business_hours'] = ((df['timestamp'].dt.hour >= 9) & 
                                  (df['timestamp'].dt.hour <= 17)).astype(int)
        return df
    
    def extract_case_features(self, df: pd.DataFrame, cases_df: pd.DataFrame) -> pd.DataFrame:
        """Merge case-level static features"""
        df = df.copy()
        
        # Define expected case columns
        case_cols = ['customer_tier', 'loan_amount', 'region', 'complexity_score']
        
        # Drop any pre-existing case columns to avoid merge suffix conflicts
        existing_case_cols = [col for col in case_cols if col in df.columns]
        if existing_case_cols:
            df = df.drop(columns=existing_case_cols)
        
        # Merge case attributes
        case_features = cases_df[['case_id'] + case_cols].copy()
        df = df.merge(case_features, on='case_id', how='left')
        
        # Verify merge was successful
        if 'customer_tier' not in df.columns:
            print(f"⚠️ Warning: Merge failed. Available columns: {df.columns.tolist()[:10]}...")
            # Fill with defaults if merge failed
            df['customer_tier'] = 'Silver'
            df['loan_amount'] = 50000
            df['region'] = 'North'
            df['complexity_score'] = 5
        
        # Encode categorical features
        df['customer_tier_encoded'] = df['customer_tier'].map({
            'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3
        }).fillna(1)  # Default to Silver if unmapped
        df['region_encoded'] = df['region'].map({
            'North': 0, 'South': 1, 'East': 2, 'West': 3
        }).fillna(0)  # Default to North if unmapped
        
        # Log transform loan amount
        df['loan_amount_log'] = np.log1p(df['loan_amount'].fillna(50000))
        
        return df
    
    def extract_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract activity and resource-based features"""
        df = df.copy()
        
        # Activity encoding
        activity_map = {
            "Submit Application": 0,
            "Check Credit": 1, 
            "Manual Review": 2,
            "Quality Assurance": 3,
            "Approve": 4,
            "Reject": 5
        }
        df['activity_encoded'] = df['activity'].map(activity_map)
        
        # Resource workload features
        resource_stats = df.groupby('resource').agg({
            'case_id': 'count',
            'timestamp': ['min', 'max']
        }).reset_index()
        resource_stats.columns = ['resource', 'total_cases', 'first_case', 'last_case']
        resource_stats['resource_experience_days'] = (
            resource_stats['last_case'] - resource_stats['first_case']
        ).dt.days + 1
        resource_stats['cases_per_day'] = (
            resource_stats['total_cases'] / resource_stats['resource_experience_days']
        )
        
        df = df.merge(resource_stats[['resource', 'cases_per_day']], 
                     on='resource', how='left')
        
        return df
    
    def extract_process_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract current process state and historical features"""
        df = df.copy()
        df = df.sort_values(['case_id', 'timestamp'])
        
        # Calculate case-level statistics up to current point
        case_stats = []
        for case_id in df['case_id'].unique():
            case_data = df[df['case_id'] == case_id].copy()
            
            for i, row in case_data.iterrows():
                current_idx = case_data.index.get_loc(i)
                past_events = case_data.iloc[:current_idx + 1]
                
                stats = {
                    'case_id': case_id,
                    'timestamp': row['timestamp'],
                    'events_so_far': len(past_events),
                    'completed_so_far': (past_events['status'] == 'COMPLETED').sum(),
                    'avg_duration_so_far': past_events['duration_hours'].mean() if 'duration_hours' in past_events.columns else 0,
                    'unique_resources_so_far': past_events['resource'].nunique(),
                    'current_position': current_idx + 1,
                    'total_estimated_activities': len(case_data)
                }
                case_stats.append(stats)
        
        case_features_df = pd.DataFrame(case_stats)
        df = df.merge(case_features_df, on=['case_id', 'timestamp'], how='left')
        
        # Progress ratio
        df['progress_ratio'] = df['current_position'] / df['total_estimated_activities']
        
        return df
    
    def extract_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract system-wide workload features"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Rolling window features (last 4 hours, 24 hours)
        windows = [4, 24]  # hours
        for window in windows:
            # Use a numeric dummy column for robust rolling counts
            # This avoids "ValueError: could not convert string to float" on string columns
            indexer = df.set_index('timestamp')
            
            # Count events (proxy for workload)
            # We use sum() on a column of 1s which is equivalent to count() but safer for rolling
            # Direct assignment avoids merge explosion on duplicate timestamps
            rolling_series = indexer.assign(load_dummy=1)['load_dummy'].rolling(f'{window}H').sum()
            df[f'cases_last_{window}h'] = rolling_series.values
            
            # Note: Removed unique activity/resource counts as they caused string conversion errors
            # and 'cases_last_Xh' (event count) is a sufficient proxy for system load
        
        # Fill NaN for early timestamps
        # Fill NaN for early timestamps
        rolling_cols = [f'cases_last_{w}h' for w in windows]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        
        return df
    
    def extract_sla_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract SLA-related features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sla_deadline'] = pd.to_datetime(df['sla_deadline'])
        
        # Time to SLA deadline
        df['hours_to_sla_deadline'] = (
            df['sla_deadline'] - df['timestamp']
        ).dt.total_seconds() / 3600
        
        # SLA urgency categories
        df['sla_urgent'] = (df['hours_to_sla_deadline'] <= 4).astype(int)
        df['sla_critical'] = (df['hours_to_sla_deadline'] <= 1).astype(int)
        
        # Historical SLA breach rate by activity/resource
        activity_breach_rate = df[df['status'] == 'COMPLETED'].groupby('activity')['will_breach_sla'].mean()
        resource_breach_rate = df[df['status'] == 'COMPLETED'].groupby('resource')['will_breach_sla'].mean()
        
        df['activity_breach_rate'] = df['activity'].map(activity_breach_rate).fillna(0)
        df['resource_breach_rate'] = df['resource'].map(resource_breach_rate).fillna(0)
        
        return df
    
    def calculate_duration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate duration-based features"""
        df = df.copy()
        # Convert timestamp to datetime first (fixes string subtraction error)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['case_id', 'timestamp'])
        
        # If duration_hours already exists in the input data (from data_generation.py),
        # use it directly instead of recalculating to avoid merge conflicts
        if 'duration_hours' in df.columns:
            df['duration_hours'] = df['duration_hours'].fillna(0)
            return df
        
        # Calculate activity durations from ASSIGNED/COMPLETED event pairs
        completed_events = df[df['status'] == 'COMPLETED'].copy()
        assigned_events = df[df['status'] == 'ASSIGNED'].copy()
        
        # Merge to get start/end times for each activity
        activity_durations = []
        for case_id in df['case_id'].unique():
            case_assigned = assigned_events[assigned_events['case_id'] == case_id]
            case_completed = completed_events[completed_events['case_id'] == case_id]
            
            for _, assigned_row in case_assigned.iterrows():
                matching_completed = case_completed[
                    (case_completed['activity'] == assigned_row['activity']) &
                    (case_completed['timestamp'] > assigned_row['timestamp'])
                ]
                
                if not matching_completed.empty:
                    completed_row = matching_completed.iloc[0]
                    duration = (
                        completed_row['timestamp'] - assigned_row['timestamp']
                    ).total_seconds() / 3600
                    
                    activity_durations.append({
                        'case_id': case_id,
                        'activity': assigned_row['activity'],
                        'timestamp': assigned_row['timestamp'],
                        'calculated_duration': duration
                    })
        
        # Use a different column name to avoid merge conflicts
        duration_df = pd.DataFrame(activity_durations)
        if not duration_df.empty:
            df = df.merge(duration_df, on=['case_id', 'activity', 'timestamp'], how='left')
            df['duration_hours'] = df['calculated_duration'].fillna(0)
            df = df.drop(columns=['calculated_duration'], errors='ignore')
        else:
            df['duration_hours'] = 0
        
        return df
    
    def create_feature_matrix(self, events_df: pd.DataFrame, cases_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create complete feature matrix for ML"""
        print("🔧 Engineering features...")
        
        # Apply all feature extraction steps
        df = events_df.copy()
        
        # Calculate durations first
        df = self.calculate_duration_features(df)
        
        # Extract all feature types
        df = self.extract_temporal_features(df)
        df = self.extract_case_features(df, cases_df)
        df = self.extract_activity_features(df)
        df = self.extract_process_state_features(df)
        df = self.extract_workload_features(df)
        df = self.extract_sla_features(df)
        
        # Define feature columns (exclude non-numeric and target columns)
        exclude_cols = [
            'event_id', 'case_id', 'activity', 'status', 'timestamp', 'resource', 
            'sla_deadline', 'case_completion_time', 'customer_tier', 'region',
            'will_breach_sla', 'remaining_time_hours'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle any remaining categorical columns
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        # Fill missing values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        self.feature_columns = feature_cols
        print(f"✅ Created {len(feature_cols)} features")
        
        return df, feature_cols

def prepare_training_data(events_path: str = 'events_log.csv', 
                         cases_path: str = 'cases_table.csv') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare training data with all features"""
    
    # Load data
    print("📊 Loading data...")
    events_df = pd.read_csv(events_path)
    cases_df = pd.read_csv(cases_path)
    
    # Initialize feature engineer
    fe = ProcessFeatureEngineer()
    
    # Create feature matrix
    df, feature_cols = fe.create_feature_matrix(events_df, cases_df)
    
    # Prepare training data (only use assigned events for prediction)
    training_data = df[df['status'] == 'ASSIGNED'].copy()
    
    X = training_data[feature_cols]
    y = training_data['will_breach_sla']
    
    print(f"🎯 Training data shape: {X.shape}")
    print(f"📈 Target distribution: {y.value_counts(normalize=True)}")
    
    return X, y, feature_cols

if __name__ == "__main__":
    # Test the feature engineering pipeline
    X, y, features = prepare_training_data()
    print(f"\n✨ Feature engineering complete!")
    print(f"📋 Features: {features}")
    print(f"\n📊 Sample features:\n{X.head()}")
