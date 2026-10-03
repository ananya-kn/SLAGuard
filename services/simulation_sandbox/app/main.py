import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from app.state_manager import StateManager
from app.simulator import ProcessSimulator, SimulationConfig
from app.scenario import ScenarioBuilder, Scenario
from app.comparator import ScenarioComparator

app = FastAPI(title="Simulation Sandbox Service")

# Enable CORS for demo UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
state_manager = StateManager()
scenario_builder = ScenarioBuilder()
comparator = ScenarioComparator()

# Pydantic models
class ScenarioRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    resource_changes: Optional[Dict[str, int]] = None
    capacity_multipliers: Optional[Dict[str, float]] = None
    arrival_rate_multiplier: float = 1.0

class SimulationRequest(BaseModel):
    scenario_id: Optional[str] = None
    horizon_hours: float = 24
    num_runs: int = 100

class CompareRequest(BaseModel):
    scenario_id: str
    horizon_hours: float = 24
    num_runs: int = 100

@app.on_event("startup")
async def startup():
    state_manager.connect()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "simulation-sandbox"}

@app.get("/state/snapshot")
def get_state_snapshot():
    """Get current WIP snapshot for simulation"""
    try:
        state = state_manager.get_current_state()
        return {
            "timestamp": state.timestamp.isoformat(),
            "active_case_count": len(state.active_cases['case_id'].unique()) if not state.active_cases.empty else 0,
            "queue_depths": state.queue_depths,
            "resource_availability": state.resource_availability
        }
    except Exception as e:
        return {"error": str(e), "active_case_count": 0, "queue_depths": {}, "resource_availability": {}}

@app.post("/scenario/create")
def create_scenario(request: ScenarioRequest):
    """Create a new what-if scenario"""
    scenario = scenario_builder.create_scenario(
        name=request.name,
        description=request.description,
        resource_changes=request.resource_changes,
        capacity_multipliers=request.capacity_multipliers,
        arrival_rate_multiplier=request.arrival_rate_multiplier
    )
    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "resource_changes": scenario.resource_changes
    }

@app.get("/scenario/list")
def list_scenarios():
    """List all created scenarios"""
    return {"scenarios": scenario_builder.list_scenarios()}

@app.get("/scenario/{scenario_id}")
def get_scenario(scenario_id: str):
    """Get scenario details"""
    scenario = scenario_builder.get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "id": scenario.id,
        "name": scenario.name,
        "description": scenario.description,
        "resource_changes": scenario.resource_changes,
        "arrival_rate_multiplier": scenario.arrival_rate_multiplier
    }

@app.post("/scenario/run")
def run_simulation(request: SimulationRequest):
    """Run simulation with optional scenario applied"""
    try:
        state = state_manager.get_current_state()
        
        # Prepare initial cases
        initial_cases = []
        if not state.active_cases.empty:
            for case_id in state.active_cases['case_id'].unique():
                case_data = state.active_cases[state.active_cases['case_id'] == case_id].iloc[-1]
                initial_cases.append({
                    "case_id": case_id,
                    "current_activity": case_data['activity'],
                    "sla_hours": 48,  # Default
                    "complexity_score": 5
                })
        
        # Apply scenario if provided
        config = SimulationConfig(
            resource_capacities=ProcessSimulator.DEFAULT_CAPACITIES.copy()
        )
        
        if request.scenario_id:
            scenario = scenario_builder.get_scenario(request.scenario_id)
            if scenario:
                config.resource_capacities = scenario_builder.apply_to_config(
                    scenario, config.resource_capacities
                )
        
        # Run simulation
        simulator = ProcessSimulator(config)
        results = simulator.run_simulation(
            initial_cases=initial_cases,
            horizon_hours=request.horizon_hours,
            num_runs=request.num_runs
        )
        
        return {
            "scenario_id": request.scenario_id,
            "horizon_hours": request.horizon_hours,
            "num_runs": request.num_runs,
            "results": asdict(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scenario/compare")
def compare_scenarios(request: CompareRequest):
    """Compare baseline vs scenario"""
    try:
        state = state_manager.get_current_state()
        
        # Prepare initial cases
        initial_cases = []
        if not state.active_cases.empty:
            for case_id in state.active_cases['case_id'].unique():
                initial_cases.append({
                    "case_id": case_id,
                    "sla_hours": 48,
                    "complexity_score": 5
                })
        
        # Run baseline
        baseline_simulator = ProcessSimulator()
        baseline_results = baseline_simulator.run_simulation(
            initial_cases=initial_cases,
            horizon_hours=request.horizon_hours,
            num_runs=request.num_runs
        )
        
        # Run scenario
        scenario = scenario_builder.get_scenario(request.scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found")
        
        scenario_config = SimulationConfig(
            resource_capacities=scenario_builder.apply_to_config(
                scenario, ProcessSimulator.DEFAULT_CAPACITIES.copy()
            )
        )
        scenario_simulator = ProcessSimulator(scenario_config)
        scenario_results = scenario_simulator.run_simulation(
            initial_cases=initial_cases,
            horizon_hours=request.horizon_hours,
            num_runs=request.num_runs
        )
        
        # Compare
        comparison = comparator.compare(baseline_results, scenario_results, scenario.name)
        
        return {
            "scenario_name": scenario.name,
            "baseline": asdict(baseline_results),
            "scenario": asdict(scenario_results),
            "comparison": asdict(comparison)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pre-built scenario templates
@app.post("/scenario/templates/reallocation")
def create_reallocation_scenario(from_pool: str, to_pool: str, count: int = 2):
    """Quick template: Resource reallocation"""
    scenario = scenario_builder.create_resource_reallocation(from_pool, to_pool, count)
    return {"id": scenario.id, "name": scenario.name}

@app.post("/scenario/templates/volume-spike")
def create_volume_spike_scenario(multiplier: float = 1.5):
    """Quick template: Volume spike"""
    scenario = scenario_builder.create_volume_spike(multiplier)
    return {"id": scenario.id, "name": scenario.name}

@app.post("/scenario/templates/capacity-increase")
def create_capacity_increase_scenario(pool: str, additional: int = 2):
    """Quick template: Add capacity"""
    scenario = scenario_builder.create_capacity_increase(pool, additional)
    return {"id": scenario.id, "name": scenario.name}
