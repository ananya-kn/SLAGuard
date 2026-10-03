import simpy
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import yaml
import os

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    resource_capacities: Dict[str, int] = field(default_factory=dict)
    activity_durations: Dict[str, tuple] = field(default_factory=dict)  # (mean, std)
    arrival_rate: float = 1.0  # cases per hour
    sick_agents_count: int = 0  # Number of agents unavailable (sick/holiday)
    shift_modifier: float = 1.0  # 1.0 = full capacity, 0.5 = night shift
    apply_concept_drift: bool = False  # Manual Review 50% slower
    
@dataclass
class SimulationResults:
    """Results from a simulation run"""
    total_cases_completed: int
    total_sla_breaches: int
    breach_rate: float
    avg_completion_time: float
    queue_depths_over_time: List[Dict[str, int]]
    bottleneck_activities: List[str]

class ProcessSimulator:
    """SimPy-based Discrete Event Simulation engine"""
    
    DEFAULT_DURATIONS = {
        "Submit Application": (0.1, 0.05),
        "Check Credit": (2.0, 1.0),
        "Manual Review": (4.0, 2.5),
        "Quality Assurance": (1.5, 0.8),
        "Approve": (0.5, 0.2),
        "Reject": (0.3, 0.1)
    }
    
    DEFAULT_CAPACITIES = {
        "Intake": 5,
        "Credit Team": 8,
        "Review Team": 10,
        "QA Team": 4,
        "Approval Team": 6
    }
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig(
            resource_capacities=self.DEFAULT_CAPACITIES.copy(),
            activity_durations=self.DEFAULT_DURATIONS.copy()
        )
        self.results = []
    
    def run_simulation(self, initial_cases: List[Dict], 
                       horizon_hours: float = 24,
                       num_runs: int = 100) -> SimulationResults:
        """Run Monte Carlo simulation"""
        all_results = []
        
        for _ in range(num_runs):
            result = self._single_run(initial_cases, horizon_hours)
            all_results.append(result)
        
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _single_run(self, initial_cases: List[Dict], horizon_hours: float) -> Dict:
        """Execute a single simulation run"""
        env = simpy.Environment()
        
        # Create resources
        resources = {
            name: simpy.Resource(env, capacity=cap)
            for name, cap in self.config.resource_capacities.items()
        }
        
        # Track metrics
        completed = []
        breaches = []
        queue_snapshots = []
        
        # Process existing cases
        for case in initial_cases:
            env.process(self._process_case(
                env, case, resources, completed, breaches
            ))
        
        # Record queue depths periodically
        env.process(self._record_queues(env, resources, queue_snapshots, horizon_hours))
        
        # Run simulation
        env.run(until=horizon_hours)
        
        return {
            "completed": len(completed),
            "breaches": len(breaches),
            "queue_snapshots": queue_snapshots
        }
    
    def _process_case(self, env, case: Dict, resources: Dict,
                      completed: List, breaches: List):
        """Simulate a single case through the process"""
        activities = self._get_activity_sequence(case)
        sla_deadline = case.get("sla_hours", 48)
        start_time = env.now
        
        for activity in activities:
            resource_name = self._get_resource_for_activity(activity)
            resource = resources.get(resource_name)
            
            if resource:
                with resource.request() as req:
                    yield req
                    # Activity duration
                    mean, std = self.config.activity_durations.get(activity, (1.0, 0.5))
                    duration = max(0.1, np.random.lognormal(np.log(mean), std))
                    yield env.timeout(duration)
        
        # Check SLA
        total_time = env.now - start_time
        completed.append({"case_id": case.get("case_id"), "time": total_time})
        if total_time > sla_deadline:
            breaches.append(case.get("case_id"))
    
    def _get_activity_sequence(self, case: Dict) -> List[str]:
        """Determine activity sequence for a case"""
        complexity = case.get("complexity_score", 5)
        tier = case.get("customer_tier", "Silver")
        
        if tier == "Platinum" or complexity < 3:
            return ["Submit Application", "Check Credit", "Approve"]
        elif complexity > 7:
            activities = ["Submit Application", "Check Credit", "Manual Review", "Quality Assurance"]
            activities.append("Reject" if np.random.random() < 0.3 else "Approve")
            return activities
        else:
            return ["Submit Application", "Check Credit", "Manual Review", "Approve"]
    
    def _get_resource_for_activity(self, activity: str) -> str:
        """Map activity to resource pool"""
        mapping = {
            "Submit Application": "Intake",
            "Check Credit": "Credit Team",
            "Manual Review": "Review Team",
            "Quality Assurance": "QA Team",
            "Approve": "Approval Team",
            "Reject": "Approval Team"
        }
        return mapping.get(activity, "Intake")
    
    def _record_queues(self, env, resources: Dict, snapshots: List, horizon: float):
        """Record queue depths at regular intervals"""
        while env.now < horizon:
            snapshot = {name: len(res.queue) for name, res in resources.items()}
            snapshots.append({"time": env.now, "queues": snapshot})
            yield env.timeout(1.0)  # Every hour
    
    def _aggregate_results(self, all_results: List[Dict]) -> SimulationResults:
        """Aggregate multiple simulation runs"""
        total_completed = sum(r["completed"] for r in all_results)
        total_breaches = sum(r["breaches"] for r in all_results)
        
        breach_rate = total_breaches / max(1, total_completed)
        
        # Find common bottlenecks
        bottlenecks = self._identify_bottlenecks(all_results)
        
        return SimulationResults(
            total_cases_completed=total_completed // len(all_results),
            total_sla_breaches=total_breaches // len(all_results),
            breach_rate=round(breach_rate, 3),
            avg_completion_time=0,  # Would need to track this
            queue_depths_over_time=[],
            bottleneck_activities=bottlenecks
        )
    
    def _identify_bottlenecks(self, all_results: List[Dict]) -> List[str]:
        """Identify consistently overloaded activities"""
        queue_totals = {}
        for result in all_results:
            for snapshot in result.get("queue_snapshots", []):
                for activity, depth in snapshot.get("queues", {}).items():
                    queue_totals[activity] = queue_totals.get(activity, 0) + depth
        
        # Return activities with highest average queue
        sorted_activities = sorted(queue_totals.items(), key=lambda x: -x[1])
        return [a[0] for a in sorted_activities[:3] if a[1] > 0]
