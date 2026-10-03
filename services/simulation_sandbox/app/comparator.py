from dataclasses import dataclass
from typing import Dict, List, Optional
from app.simulator import SimulationResults

@dataclass
class ComparisonResult:
    """Result of comparing two scenarios"""
    baseline_breach_rate: float
    scenario_breach_rate: float
    breach_rate_change: float  # Negative is improvement
    baseline_bottlenecks: List[str]
    scenario_bottlenecks: List[str]
    recommendation: str

class ScenarioComparator:
    """Compares baseline vs modified scenario results"""
    
    def compare(self, baseline: SimulationResults, 
                scenario: SimulationResults,
                scenario_name: str = "") -> ComparisonResult:
        """Compare baseline and scenario results"""
        
        breach_change = scenario.breach_rate - baseline.breach_rate
        breach_pct_change = (breach_change / max(0.001, baseline.breach_rate)) * 100
        
        recommendation = self._generate_recommendation(
            baseline, scenario, breach_change, scenario_name
        )
        
        return ComparisonResult(
            baseline_breach_rate=baseline.breach_rate,
            scenario_breach_rate=scenario.breach_rate,
            breach_rate_change=round(breach_change, 3),
            baseline_bottlenecks=baseline.bottleneck_activities,
            scenario_bottlenecks=scenario.bottleneck_activities,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, baseline: SimulationResults,
                                   scenario: SimulationResults,
                                   breach_change: float,
                                   scenario_name: str) -> str:
        """Generate actionable recommendation"""
        
        if breach_change < -0.1:
            return f"✅ RECOMMENDED: '{scenario_name}' reduces SLA breaches by {abs(breach_change)*100:.1f}%"
        elif breach_change < 0:
            return f"⚠️ Marginal improvement: '{scenario_name}' reduces breaches by {abs(breach_change)*100:.1f}%"
        elif breach_change > 0.1:
            return f"❌ NOT RECOMMENDED: '{scenario_name}' increases SLA breaches by {breach_change*100:.1f}%"
        else:
            return f"➡️ Neutral: '{scenario_name}' has minimal impact on SLA performance"
    
    def compare_multiple(self, baseline: SimulationResults,
                          scenarios: Dict[str, SimulationResults]) -> List[Dict]:
        """Compare baseline against multiple scenarios"""
        comparisons = []
        
        for name, result in scenarios.items():
            comparison = self.compare(baseline, result, name)
            comparisons.append({
                "scenario_name": name,
                "breach_rate_change": comparison.breach_rate_change,
                "recommendation": comparison.recommendation
            })
        
        # Sort by improvement (most negative breach change first)
        return sorted(comparisons, key=lambda x: x["breach_rate_change"])
