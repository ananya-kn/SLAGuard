"""
Comprehensive Feature Preprocessor for Inference Service
Replicates ALL feature engineering from Model_Train for consistent predictions

Features by model:
- Duration: 26 features (activity, temporal, complexity, workload)
- Routing: 17 features (activity sequences, context, process state)
- SLA: 41 features (full feature engineering + SLA-specific)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class FeaturePreprocessor:
    """Replicates training feature engineering for inference"""
    
    # ========== COMMON MAPPINGS ==========
    ACTIVITY_MAP = {
        "Submit Application": 0, "Check Credit": 1, "Manual Review": 2,
        "Quality Assurance": 3, "Approve": 4, "Reject": 5
    }
    
    TIER_MAP = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    REGION_MAP = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
    
    # Activity base durations (from process config)
    ACTIVITY_BASE_DURATION = {
        "Submit Application": 0.1, "Check Credit": 2.0, "Manual Review": 4.0,
        "Quality Assurance": 1.5, "Approve": 0.5, "Reject": 0.3
    }
    
    TIER_SPEED_MODIFIER = {'Bronze': 1.2, 'Silver': 1.1, 'Gold': 1.0, 'Platinum': 0.8}
    
    # Historical breach rates (learned defaults, would normally come from DB)
    ACTIVITY_BREACH_RATE = {
        "Submit Application": 0.05, "Check Credit": 0.15, "Manual Review": 0.35,
        "Quality Assurance": 0.20, "Approve": 0.02, "Reject": 0.01
    }
    
    RESOURCE_BREACH_RATE_DEFAULT = 0.15
    
    # ========== DURATION MODEL FEATURES (26 features) ==========
    DURATION_FEATURES = [
        'activity_encoded', 'tier_encoded', 'region_encoded', 'complexity_score', 'priority',
        'assigned_hour', 'assigned_day_of_week', 'assigned_month',
        'is_weekend', 'is_business_hours', 'is_peak_hour', 'is_night_shift',
        'is_monday', 'is_friday',
        'activity_base_duration', 'expected_duration', 'complexity_duration_modifier',
        'workload_proxy', 'tier_speed_modifier',
        'resource_task_count', 'resource_activity_variety', 'activity_frequency',
        'complexity_x_activity', 'loan_amount_log', 'priority_x_complexity', 'expected_x_workload'
    ]
    
    # ========== ROUTING MODEL FEATURES (17 features) ==========
    ROUTING_FEATURES = [
        'current_activity_encoded', 'tier_encoded', 'region_encoded', 'complexity_score',
        'position_in_process', 'progress_ratio', 'priority',
        'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_business_hours',
        'is_start_of_process', 'resource_sample_count',
        'complexity_x_position', 'loan_amount_log', 'prev_activity_count'
    ]
    
    # ========== SLA MODEL FEATURES (41 features) ==========
    # These match the training output from feature_engineering.py + create_sla_features
    SLA_FEATURES = [
        # From feature_engineering.py base features
        'priority', 'hour_of_day', 'day_of_week', 'available_resources_count',
        'queue_depth_estimate', 'duration_hours', 'day_of_month', 'month',
        'is_weekend', 'is_business_hours', 'loan_amount', 'complexity_score',
        'customer_tier_encoded', 'region_encoded', 'loan_amount_log', 'activity_encoded',
        'cases_per_day',
        # Process state features
        'events_so_far', 'completed_so_far', 'avg_duration_so_far',
        'unique_resources_so_far', 'current_position', 'total_estimated_activities',
        'progress_ratio',
        # Workload features  
        'cases_last_4h', 'cases_last_24h',
        # SLA-specific features
        'hours_to_sla_deadline', 'sla_urgent', 'sla_critical',
        'activity_breach_rate', 'resource_breach_rate',
        'hours_to_sla', 'sla_pressure_ratio',
        'resource_workload', 'activity_workload',
        'complexity_x_time_pressure', 'priority_x_urgency',
        'activity_historical_risk', 'resource_historical_risk',
        'total_active_cases', 'system_load_ratio'
    ]

    @classmethod
    def preprocess_duration(cls, raw_input: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for duration prediction - 26 features"""
        
        activity = raw_input.get('activity', 'Manual Review')
        customer_tier = raw_input.get('customer_tier', 'Silver')
        complexity_score = float(raw_input.get('complexity_score', 5.0))
        region = raw_input.get('region', 'North')
        loan_amount = float(raw_input.get('loan_amount', 50000))
        priority = int(raw_input.get('priority', 3))
        
        now = datetime.now()
        assigned_hour = int(raw_input.get('assigned_hour', now.hour))
        assigned_day_of_week = int(raw_input.get('assigned_day_of_week', now.weekday()))
        assigned_month = int(raw_input.get('assigned_month', now.month))
        
        activity_encoded = cls.ACTIVITY_MAP.get(activity, 2)
        tier_encoded = cls.TIER_MAP.get(customer_tier, 1)
        region_encoded = cls.REGION_MAP.get(region, 0)
        
        is_weekend = 1 if assigned_day_of_week >= 5 else 0
        is_business_hours = 1 if 9 <= assigned_hour <= 17 else 0
        is_peak_hour = 1 if 10 <= assigned_hour <= 15 else 0
        is_night_shift = 1 if assigned_hour < 6 or assigned_hour > 22 else 0
        is_monday = 1 if assigned_day_of_week == 0 else 0
        is_friday = 1 if assigned_day_of_week == 4 else 0
        
        activity_base_duration = cls.ACTIVITY_BASE_DURATION.get(activity, 2.0)
        complexity_duration_modifier = 1.0 + (complexity_score - 5) * 0.1
        expected_duration = activity_base_duration * complexity_duration_modifier
        
        if 9 <= assigned_hour <= 11:
            workload_proxy = 1.3
        elif 14 <= assigned_hour <= 16:
            workload_proxy = 1.2
        elif assigned_hour < 7 or assigned_hour > 19:
            workload_proxy = 0.8
        else:
            workload_proxy = 1.0
        
        tier_speed_modifier = cls.TIER_SPEED_MODIFIER.get(customer_tier, 1.0)
        
        resource_task_count = float(raw_input.get('resource_task_count', 100))
        resource_activity_variety = float(raw_input.get('resource_activity_variety', 5))
        activity_frequency = float(raw_input.get('activity_frequency', 500))
        
        complexity_x_activity = complexity_score * activity_encoded
        loan_amount_log = np.log1p(loan_amount)
        priority_x_complexity = priority * complexity_score
        expected_x_workload = expected_duration * workload_proxy
        
        features = [
            activity_encoded, tier_encoded, region_encoded, complexity_score, priority,
            assigned_hour, assigned_day_of_week, assigned_month,
            is_weekend, is_business_hours, is_peak_hour, is_night_shift,
            is_monday, is_friday,
            activity_base_duration, expected_duration, complexity_duration_modifier,
            workload_proxy, tier_speed_modifier,
            resource_task_count, resource_activity_variety, activity_frequency,
            complexity_x_activity, loan_amount_log, priority_x_complexity, expected_x_workload
        ]
        
        return np.array([features], dtype=np.float32)
    
    @classmethod
    def preprocess_routing(cls, raw_input: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for routing prediction - 17 features"""
        
        activity = raw_input.get('current_activity', raw_input.get('activity', 'Manual Review'))
        customer_tier = raw_input.get('customer_tier', 'Silver')
        complexity_score = float(raw_input.get('complexity_score', 5.0))
        region = raw_input.get('region', 'North')
        loan_amount = float(raw_input.get('loan_amount', 50000))
        priority = int(raw_input.get('priority', 3))
        position_in_process = int(raw_input.get('position_in_process', 1))
        progress_ratio = float(raw_input.get('progress_ratio', 0.5))
        prev_activity_count = int(raw_input.get('prev_activity_count', 1))
        
        now = datetime.now()
        hour_of_day = int(raw_input.get('hour_of_day', now.hour))
        day_of_week = int(raw_input.get('day_of_week', now.weekday()))
        month = int(raw_input.get('month', now.month))
        
        current_activity_encoded = cls.ACTIVITY_MAP.get(activity, 2)
        tier_encoded = cls.TIER_MAP.get(customer_tier, 1)
        region_encoded = cls.REGION_MAP.get(region, 0)
        
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if 9 <= hour_of_day <= 17 else 0
        is_start_of_process = 1 if position_in_process == 1 else 0
        resource_sample_count = float(raw_input.get('resource_sample_count', 100))
        complexity_x_position = complexity_score * position_in_process
        loan_amount_log = np.log1p(loan_amount)
        
        features = [
            current_activity_encoded, tier_encoded, region_encoded, complexity_score,
            position_in_process, progress_ratio, priority,
            hour_of_day, day_of_week, month, is_weekend, is_business_hours,
            is_start_of_process, resource_sample_count,
            complexity_x_position, loan_amount_log, prev_activity_count
        ]
        
        return np.array([features], dtype=np.float32)
    
    @classmethod
    def preprocess_sla_breach(cls, raw_input: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess features for SLA breach prediction - 41 features
        Replicates feature_engineering.py + create_sla_features from training
        """
        
        # Extract base inputs
        activity = raw_input.get('activity', 'Manual Review')
        customer_tier = raw_input.get('customer_tier', 'Silver')
        complexity_score = float(raw_input.get('complexity_score', 5.0))
        region = raw_input.get('region', 'North')
        loan_amount = float(raw_input.get('loan_amount', 50000))
        priority = int(raw_input.get('priority', 3))
        resource = raw_input.get('resource', 'Agent_001')
        
        # SLA-specific inputs
        hours_to_sla = float(raw_input.get('hours_to_sla', raw_input.get('hours_to_sla_deadline', 24)))
        elapsed_time = float(raw_input.get('elapsed_time', 0))
        duration_hours = float(raw_input.get('duration_hours', 0))
        
        # Process state inputs (with defaults for inference)
        events_so_far = int(raw_input.get('events_so_far', 2))
        completed_so_far = int(raw_input.get('completed_so_far', 1))
        current_position = int(raw_input.get('current_position', 2))
        total_estimated_activities = int(raw_input.get('total_estimated_activities', 6))
        unique_resources_so_far = int(raw_input.get('unique_resources_so_far', 1))
        avg_duration_so_far = float(raw_input.get('avg_duration_so_far', 2.0))
        
        # Workload inputs (with defaults)
        available_resources_count = int(raw_input.get('available_resources_count', 15))
        queue_depth_estimate = int(raw_input.get('queue_depth_estimate', 5))
        cases_last_4h = int(raw_input.get('cases_last_4h', 20))
        cases_last_24h = int(raw_input.get('cases_last_24h', 100))
        total_active_cases = int(raw_input.get('total_active_cases', 50))
        
        # Temporal features
        now = datetime.now()
        hour_of_day = int(raw_input.get('hour_of_day', now.hour))
        day_of_week = int(raw_input.get('day_of_week', now.weekday()))
        day_of_month = int(raw_input.get('day_of_month', now.day))
        month = int(raw_input.get('month', now.month))
        
        # Encode categoricals
        activity_encoded = cls.ACTIVITY_MAP.get(activity, 2)
        customer_tier_encoded = cls.TIER_MAP.get(customer_tier, 1)
        region_encoded = cls.REGION_MAP.get(region, 0)
        
        # Temporal flags
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if 9 <= hour_of_day <= 17 else 0
        
        # Derived features
        loan_amount_log = np.log1p(loan_amount)
        cases_per_day = float(raw_input.get('cases_per_day', 10.0))
        progress_ratio = current_position / max(total_estimated_activities, 1)
        
        # SLA-specific features
        hours_to_sla_deadline = hours_to_sla
        sla_urgent = 1 if hours_to_sla <= 4 else 0
        sla_critical = 1 if hours_to_sla <= 1 else 0
        sla_pressure_ratio = hours_to_sla / 24.0
        
        # Historical breach rates (from lookup tables or defaults)
        activity_breach_rate = cls.ACTIVITY_BREACH_RATE.get(activity, 0.15)
        resource_breach_rate = float(raw_input.get('resource_breach_rate', cls.RESOURCE_BREACH_RATE_DEFAULT))
        
        # Workload features
        resource_workload = float(raw_input.get('resource_workload', 5))
        activity_workload = float(raw_input.get('activity_workload', 20))
        
        # Interaction features
        complexity_x_time_pressure = complexity_score * (1 / (hours_to_sla + 1))
        priority_x_urgency = priority * sla_urgent
        
        # Historical risk (same as breach rates in simple case)
        activity_historical_risk = activity_breach_rate
        resource_historical_risk = resource_breach_rate
        
        # System load
        system_load_ratio = resource_workload / max(total_active_cases, 1)
        
        # Build feature array in EXACT order matching training
        features = [
            # Base features from feature_engineering.py
            priority, hour_of_day, day_of_week, available_resources_count,
            queue_depth_estimate, duration_hours, day_of_month, month,
            is_weekend, is_business_hours, loan_amount, complexity_score,
            customer_tier_encoded, region_encoded, loan_amount_log, activity_encoded,
            cases_per_day,
            # Process state features
            events_so_far, completed_so_far, avg_duration_so_far,
            unique_resources_so_far, current_position, total_estimated_activities,
            progress_ratio,
            # Workload features
            cases_last_4h, cases_last_24h,
            # SLA-specific features
            hours_to_sla_deadline, sla_urgent, sla_critical,
            activity_breach_rate, resource_breach_rate,
            hours_to_sla, sla_pressure_ratio,
            resource_workload, activity_workload,
            complexity_x_time_pressure, priority_x_urgency,
            activity_historical_risk, resource_historical_risk,
            total_active_cases, system_load_ratio
        ]
        
        return np.array([features], dtype=np.float32)
    
    @classmethod
    def get_feature_names(cls, model_type: str) -> List[str]:
        """Get feature names for a given model type"""
        if 'duration' in model_type.lower():
            return cls.DURATION_FEATURES
        elif 'routing' in model_type.lower():
            return cls.ROUTING_FEATURES
        elif 'sla' in model_type.lower() or 'breach' in model_type.lower():
            return cls.SLA_FEATURES
        else:
            return cls.DURATION_FEATURES
