from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

@dataclass
class Scenario:
    """Defines a what-if scenario for testing"""
    id: str
    name: str
    description: str = ""
    resource_changes: Dict[str, int] = field(default_factory=dict)  # {"Review Team": +3}
    capacity_multipliers: Dict[str, float] = field(default_factory=dict)  # {"Manual Review": 1.5}
    arrival_rate_multiplier: float = 1.0  # Simulate volume spike
    sick_agents_count: int = 0  # "3 agents sick" scenario
    shift_modifier: float = 1.0  # Night shift = 0.5
    created_at: datetime = field(default_factory=datetime.now)

class ScenarioBuilder:
    """Builds and manages what-if scenarios"""
    
    def __init__(self):
        self.scenarios: Dict[str, Scenario] = {}
    
    def create_scenario(self, name: str, 
                        resource_changes: Optional[Dict[str, int]] = None,
                        capacity_multipliers: Optional[Dict[str, float]] = None,
                        arrival_rate_multiplier: float = 1.0,
                        description: str = "") -> Scenario:
        """Create a new what-if scenario"""
        scenario = Scenario(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            resource_changes=resource_changes or {},
            capacity_multipliers=capacity_multipliers or {},
            arrival_rate_multiplier=arrival_rate_multiplier
        )
        self.scenarios[scenario.id] = scenario
        return scenario
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Retrieve a scenario by ID"""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self) -> List[Dict]:
        """List all scenarios"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "resource_changes": s.resource_changes,
                "created_at": s.created_at.isoformat()
            }
            for s in self.scenarios.values()
        ]
    
    def apply_to_config(self, scenario: Scenario, base_capacities: Dict[str, int]) -> Dict[str, int]:
        """Apply scenario changes to base configuration"""
        modified = base_capacities.copy()
        
        for resource, change in scenario.resource_changes.items():
            if resource in modified:
                modified[resource] = max(1, modified[resource] + change)
        
        return modified
    
    # Pre-built scenario templates
    def create_resource_reallocation(self, from_pool: str, to_pool: str, count: int) -> Scenario:
        """Create a resource reallocation scenario"""
        return self.create_scenario(
            name=f"Move {count} from {from_pool} to {to_pool}",
            resource_changes={from_pool: -count, to_pool: count},
            description=f"Reallocate {count} resources from {from_pool} to {to_pool}"
        )
    
    def create_volume_spike(self, multiplier: float) -> Scenario:
        """Create a volume spike scenario"""
        return self.create_scenario(
            name=f"{int(multiplier*100)}% Volume Spike",
            arrival_rate_multiplier=multiplier,
            description=f"Simulate {int((multiplier-1)*100)}% increase in incoming cases"
        )
    
    def create_capacity_increase(self, pool: str, additional: int) -> Scenario:
        """Create a capacity increase scenario"""
        return self.create_scenario(
            name=f"Add {additional} to {pool}",
            resource_changes={pool: additional},
            description=f"Add {additional} resources to {pool}"
        )
