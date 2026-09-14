"""
Multi-Model Inference Engine
Integrates Duration, Routing, and SLA breach models for comprehensive predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ComprehensivePrediction:
    """Container for comprehensive prediction results"""
    case_id: str
    current_activity: str
    timestamp: datetime
    
    # Duration predictions
    predicted_duration_hours: float
    duration_confidence: float
    
    # Routing predictions
    next_activity: str
    next_activity_probability: float
    routing_confidence: float
    
    # SLA predictions
    sla_breach_probability: float
    sla_risk_level: str
    hours_to_sla: float
    
    # Overall assessment
    overall_risk_score: float
    recommendations: List[str]

class MultiModelInferenceEngine:
    def __init__(self, 
                 duration_model_path: str = 'duration_model.json',
                 routing_model_path: str = 'routing_model.json',
                 sla_model_path: str = 'sla_model.json'):
        """Initialize multi-model inference engine"""
        
        self.duration_model = None
        self.routing_model = None
        self.sla_model = None
        
        self.duration_metadata = None
        self.routing_metadata = None
        self.sla_metadata = None
        
        self.routing_label_encoder = None
        
        self.load_all_models(duration_model_path, routing_model_path, sla_model_path)
    
    def load_all_models(self, duration_path: str, routing_path: str, sla_path: str):
        """Load all three models"""
        
        print("🔄 Loading models...")
        
        # Load Duration Model
        try:
            self.duration_model = xgb.Booster()
            self.duration_model.load_model(duration_path)
            self.duration_metadata = joblib.load(duration_path.replace('.json', '_metadata.pkl'))
            print(f"✅ Duration model loaded from {duration_path}")
        except Exception as e:
            print(f"⚠️ Could not load duration model: {e}")
        
        # Load Routing Model
        try:
            self.routing_model = xgb.Booster()
            self.routing_model.load_model(routing_path)
            self.routing_metadata = joblib.load(routing_path.replace('.json', '_metadata.pkl'))
            
            # Reconstruct label encoder
            from sklearn.preprocessing import LabelEncoder
            self.routing_label_encoder = LabelEncoder()
            self.routing_label_encoder.classes_ = np.array(self.routing_metadata['label_encoder_classes'])
            
            print(f"✅ Routing model loaded from {routing_path}")
        except Exception as e:
            print(f"⚠️ Could not load routing model: {e}")
        
        # Load SLA Model
        try:
            self.sla_model = xgb.Booster()
            self.sla_model.load_model(sla_path)
            self.sla_metadata = joblib.load(sla_path.replace('.json', '_metadata.pkl'))
            print(f"✅ SLA model loaded from {sla_path}")
        except Exception as e:
            print(f"⚠️ Could not load SLA model: {e}")
    
    def prepare_duration_features(self, event_data: Dict, case_data: Dict) -> pd.DataFrame:
        """Prepare features for duration prediction"""
        
        # Create feature dictionary
        features = {}
        
        # Activity encoding
        activity_map = {
            "Submit Application": 0, "Check Credit": 1, "Manual Review": 2,
            "Quality Assurance": 3, "Approve": 4, "Reject": 5
        }
        tier_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
        
        # Basic features
        features['activity_encoded'] = activity_map.get(event_data['activity'], 0)
        features['tier_encoded'] = tier_map.get(case_data['customer_tier'], 0)
        features['region_encoded'] = region_map.get(case_data['region'], 0)
        features['complexity_score'] = case_data['complexity_score']
        features['priority'] = event_data.get('priority', 3)
        
        # Time features
        timestamp = pd.to_datetime(event_data['timestamp'])
        features['assigned_hour'] = timestamp.hour
        features['assigned_day_of_week'] = timestamp.dayofweek
        features['assigned_month'] = timestamp.month
        
        # Placeholder values for resource and activity stats (would be calculated from historical data)
        features['avg_duration'] = 2.0  # Default average
        features['total_tasks'] = 50
        features['duration_std'] = 1.0
        features['unique_activities'] = 3
        features['activity_avg_duration'] = 2.0
        features['activity_std_duration'] = 1.0
        features['activity_count'] = 100
        
        # Boolean features
        features['is_weekend'] = int(timestamp.dayofweek >= 5)
        features['is_business_hours'] = int(9 <= timestamp.hour <= 17)
        features['is_peak_hour'] = int(10 <= timestamp.hour <= 15)
        
        # Interaction features
        features['complexity_x_activity'] = features['complexity_score'] * features['activity_encoded']
        features['loan_amount_log'] = np.log1p(case_data['loan_amount'])
        
        return pd.DataFrame([features])
    
    def prepare_routing_features(self, event_data: Dict, case_data: Dict) -> pd.DataFrame:
        """Prepare features for routing prediction"""
        
        features = {}
        
        # Activity encoding
        current_activity_map = {
            "Submit Application": 0, "Check Credit": 1, "Manual Review": 2,
            "Quality Assurance": 3, "Approve": 4, "Reject": 5
        }
        tier_map = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        region_map = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
        
        # Basic features
        features['current_activity_encoded'] = current_activity_map.get(event_data['activity'], 0)
        features['tier_encoded'] = tier_map.get(case_data['customer_tier'], 0)
        features['region_encoded'] = region_map.get(case_data['region'], 0)
        features['complexity_score'] = case_data['complexity_score']
        features['priority'] = event_data.get('priority', 3)
        
        # Time features
        timestamp = pd.to_datetime(event_data['timestamp'])
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.dayofweek
        features['month'] = timestamp.month
        features['is_weekend'] = int(timestamp.dayofweek >= 5)
        features['is_business_hours'] = int(9 <= timestamp.hour <= 17)
        
        # Process context (simplified)
        features['position_in_process'] = 2  # Would be calculated from actual process position
        features['total_activities'] = 4
        features['progress_ratio'] = 0.5
        features['activities_remaining'] = 2
        
        # Placeholder resource and transition features
        features['total_processed'] = 50
        features['unique_next_activities'] = 3
        features['transition_probability'] = 0.3
        
        # Interaction features
        features['complexity_x_position'] = features['complexity_score'] * features['position_in_process']
        features['loan_amount_log'] = np.log1p(case_data['loan_amount'])
        features['prev_activity_count'] = 1
        
        # Boolean features
        features['is_start_of_process'] = int(features['position_in_process'] == 1)
        features['is_end_of_process'] = 0
        
        return pd.DataFrame([features])
    
    def prepare_sla_features(self, event_data: Dict, case_data: Dict) -> pd.DataFrame:
        """Prepare features for SLA prediction"""
        
        # Use the existing feature engineering pipeline
        from feature_engineering import ProcessFeatureEngineer
        
        # Create temporary DataFrames
        event_df = pd.DataFrame([event_data])
        case_df = pd.DataFrame([case_data])
        
        # Use the feature engineer to create base features
        fe = ProcessFeatureEngineer()
        full_df, _ = fe.create_feature_matrix(event_df, case_df)
        
        # Extract only assigned events
        prediction_data = full_df[full_df['status'] == 'ASSIGNED'].copy()
        
        # Add SLA-specific features
        prediction_data['timestamp'] = pd.to_datetime(prediction_data['timestamp'])
        prediction_data['sla_deadline'] = pd.to_datetime(prediction_data['sla_deadline'])
        
        prediction_data['hours_to_sla'] = (
            prediction_data['sla_deadline'] - prediction_data['timestamp']
        ).dt.total_seconds() / 3600
        
        prediction_data['sla_urgent'] = (prediction_data['hours_to_sla'] <= 4).astype(int)
        prediction_data['sla_critical'] = (prediction_data['hours_to_sla'] <= 1).astype(int)
        prediction_data['sla_pressure_ratio'] = prediction_data['hours_to_sla'] / 24
        
        # Placeholder workload features
        prediction_data['resource_workload'] = 5
        prediction_data['activity_workload'] = 10
        prediction_data['complexity_x_time_pressure'] = (
            prediction_data['complexity_score'] * (1 / (prediction_data['hours_to_sla'] + 1))
        )
        prediction_data['priority_x_urgency'] = prediction_data['priority'] * prediction_data['sla_urgent']
        prediction_data['activity_historical_risk'] = 0.2
        prediction_data['resource_historical_risk'] = 0.15
        prediction_data['total_active_cases'] = 50
        prediction_data['system_load_ratio'] = 0.1
        
        return prediction_data[self.sla_metadata['feature_names']]
    
    def predict_duration(self, event_data: Dict, case_data: Dict) -> tuple[float, float]:
        """Predict activity duration"""
        if self.duration_model is None:
            return 2.0, 0.5  # Default values
        
        try:
            X = self.prepare_duration_features(event_data, case_data)
            dmatrix = xgb.DMatrix(X, feature_names=self.duration_metadata['feature_names'])
            
            prediction = float(self.duration_model.predict(dmatrix)[0])
            confidence = min(0.9, max(0.1, 1.0 - abs(prediction - 2.0) / 5.0))  # Simple confidence calc
            
            return max(0.1, prediction), confidence
        except Exception as e:
            print(f"Duration prediction error: {e}")
            return 2.0, 0.5
    
    def predict_routing(self, event_data: Dict, case_data: Dict) -> tuple[str, float, float]:
        """Predict next activity"""
        if self.routing_model is None:
            return "Check Credit", 0.5, 0.5  # Default values
        
        try:
            X = self.prepare_routing_features(event_data, case_data)
            dmatrix = xgb.DMatrix(X, feature_names=self.routing_metadata['feature_names'])
            
            probabilities = self.routing_model.predict(dmatrix)[0]
            predicted_class = np.argmax(probabilities)
            predicted_activity = self.routing_label_encoder.inverse_transform([predicted_class])[0]
            confidence = float(np.max(probabilities))
            
            return predicted_activity, confidence, float(probabilities[predicted_class])
        except Exception as e:
            print(f"Routing prediction error: {e}")
            return "Check Credit", 0.5, 0.5
    
    def predict_sla(self, event_data: Dict, case_data: Dict) -> tuple[float, str, float]:
        """Predict SLA breach"""
        if self.sla_model is None:
            return 0.3, "MEDIUM", 4.0  # Default values
        
        try:
            X = self.prepare_sla_features(event_data, case_data)
            dmatrix = xgb.DMatrix(X, feature_names=self.sla_metadata['feature_names'])
            
            breach_probability = float(self.sla_model.predict(dmatrix)[0])
            
            # Determine risk level
            if breach_probability >= 0.8:
                risk_level = "CRITICAL"
            elif breach_probability >= 0.6:
                risk_level = "HIGH"
            elif breach_probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Calculate hours to SLA
            timestamp = pd.to_datetime(event_data['timestamp'])
            sla_deadline = pd.to_datetime(event_data['sla_deadline'])
            hours_to_sla = (sla_deadline - timestamp).total_seconds() / 3600
            
            return breach_probability, risk_level, hours_to_sla
        except Exception as e:
            print(f"SLA prediction error: {e}")
            return 0.3, "MEDIUM", 4.0
    
    def predict_comprehensive(self, event_data: Dict, case_data: Dict) -> ComprehensivePrediction:
        """Make comprehensive prediction using all models"""
        
        # Get individual predictions
        duration, duration_conf = self.predict_duration(event_data, case_data)
        next_activity, routing_conf, next_prob = self.predict_routing(event_data, case_data)
        sla_prob, risk_level, hours_to_sla = self.predict_sla(event_data, case_data)
        
        # Calculate overall risk score (0-1)
        overall_risk = (
            sla_prob * 0.5 +                    # SLA risk weighted most
            (1 - routing_conf) * 0.2 +        # Routing uncertainty
            (duration / 10.0) * 0.2 +         # Duration risk
            (1 - duration_conf) * 0.1         # Duration uncertainty
        )
        overall_risk = min(1.0, max(0.0, overall_risk))
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            duration, next_activity, sla_prob, risk_level, hours_to_sla
        )
        
        return ComprehensivePrediction(
            case_id=event_data['case_id'],
            current_activity=event_data['activity'],
            timestamp=pd.to_datetime(event_data['timestamp']),
            predicted_duration_hours=duration,
            duration_confidence=duration_conf,
            next_activity=next_activity,
            next_activity_probability=next_prob,
            routing_confidence=routing_conf,
            sla_breach_probability=sla_prob,
            sla_risk_level=risk_level,
            hours_to_sla=hours_to_sla,
            overall_risk_score=overall_risk,
            recommendations=recommendations
        )
    
    def generate_recommendations(self, duration: float, next_activity: str, 
                               sla_prob: float, risk_level: str, hours_to_sla: float) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # SLA-based recommendations
        if risk_level == "CRITICAL":
            recommendations.append(
                f"🚨 CRITICAL: {sla_prob:.1%} SLA breach risk! Immediate escalation required."
            )
        elif risk_level == "HIGH":
            recommendations.append(
                f"⚠️ HIGH RISK: {sla_prob:.1%} breach probability. Consider expedited processing."
            )
        
        # Time-based recommendations
        if hours_to_sla < 2:
            recommendations.append(
                f"⏰ URGENT: Only {hours_to_sla:.1f} hours remaining to SLA deadline."
            )
        
        # Duration-based recommendations
        if duration > 4:
            recommendations.append(
                f"🕐 LONG DURATION: Expected {duration:.1f}h processing time. Consider resource allocation."
            )
        
        # Activity-specific recommendations
        activity_recommendations = {
            "Manual Review": "Consider parallel review or automated pre-screening",
            "Check Credit": "Optimize API calls or implement parallel verification",
            "Quality Assurance": "Risk-based QA sampling for efficiency",
            "Approve": "Consider auto-approval for low-risk cases"
        }
        
        if next_activity in activity_recommendations:
            recommendations.append(
                f"🎯 NEXT ACTIVITY: {activity_recommendations[next_activity]}"
            )
        
        # Routing confidence recommendations
        if next_activity == "Reject" and sla_prob > 0.5:
            recommendations.append(
                "💡 Early rejection predicted - could save processing time"
            )
        
        return recommendations if recommendations else ["✅ Processing within normal parameters"]
    
    def predict_batch(self, events_df: pd.DataFrame, cases_df: pd.DataFrame) -> List[ComprehensivePrediction]:
        """Predict for multiple events"""
        
        predictions = []
        
        for _, event_row in events_df.iterrows():
            case_row = cases_df[cases_df['case_id'] == event_row['case_id']].iloc[0]
            
            event_data = event_row.to_dict()
            case_data = case_row.to_dict()
            
            prediction = self.predict_comprehensive(event_data, case_data)
            predictions.append(prediction)
        
        return predictions

# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-model engine
    engine = MultiModelInferenceEngine()
    
    # Test with sample data
    try:
        events_df = pd.read_csv('events_log.csv')
        cases_df = pd.read_csv('cases_table.csv')
        
        # Get current assigned events
        current_events = events_df[events_df['status'] == 'ASSIGNED'].head(3)
        
        if not current_events.empty:
            print("🔮 Testing multi-model predictions...")
            
            for _, event_row in current_events.iterrows():
                case_row = cases_df[cases_df['case_id'] == event_row['case_id']].iloc[0]
                
                event_data = event_row.to_dict()
                case_data = case_row.to_dict()
                
                prediction = engine.predict_comprehensive(event_data, case_data)
                
                print(f"\n📋 Case {prediction.case_id} - {prediction.current_activity}")
                print(f"   ⏱️ Duration: {prediction.predicted_duration_hours:.2f}h (confidence: {prediction.duration_confidence:.2f})")
                print(f"   🧭 Next Activity: {prediction.next_activity} (prob: {prediction.next_activity_probability:.2f})")
                print(f"   ⚠️ SLA Risk: {prediction.sla_risk_level} ({prediction.sla_breach_probability:.1%})")
                print(f"   🎯 Overall Risk: {prediction.overall_risk_score:.2f}")
                print(f"   💡 Top Recommendation: {prediction.recommendations[0] if prediction.recommendations else 'None'}")
        
    except FileNotFoundError:
        print("⚠️ Test data not found. Please run data generation and model training first.")
