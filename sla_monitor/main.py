import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio

from app.monitor import WIPMonitor
from app.forecaster import SLAForecaster
from app.bottleneck import BottleneckDetector
from app.metrics import get_metrics, get_content_type, update_queue_metrics, update_utilization_metrics
from app.consumer import EventConsumer

app = FastAPI(title="SLA Monitor Service")

# Enable CORS for demo UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
monitor = WIPMonitor()
forecaster = SLAForecaster()
bottleneck_detector = BottleneckDetector()
event_consumer = EventConsumer()

@app.on_event("startup")
async def startup():
    monitor.connect()
    event_consumer.start()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "sla-monitor"}

@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=get_metrics(), media_type=get_content_type())

@app.get("/breaches/forecast")
async def forecast_breaches(hours: int = 24):
    """Predict SLA breaches within the specified horizon"""
    active_cases = monitor.get_active_cases()
    if active_cases.empty:
        return {"breaches": [], "message": "No active cases"}
    
    breaches = await forecaster.forecast_breaches(active_cases, hours)
    return {
        "horizon_hours": hours,
        "total_active_cases": len(active_cases['case_id'].unique()),
        "predicted_breaches": len(breaches),
        "breaches": breaches
    }

@app.get("/bottlenecks/current")
def get_current_bottlenecks():
    """Identify current bottlenecks"""
    queue_depths = monitor.get_queue_depths()
    update_queue_metrics(queue_depths)
    
    bottlenecks = bottleneck_detector.detect_bottlenecks(queue_depths)
    return {
        "queue_depths": queue_depths,
        "bottlenecks": bottlenecks
    }

@app.get("/alerts/active")
async def get_active_alerts():
    """Get all active alerts (breaches + bottlenecks)"""
    # Get breaches
    active_cases = monitor.get_active_cases()
    breaches = []
    if not active_cases.empty:
        breaches = await forecaster.forecast_breaches(active_cases, horizon_hours=4)
    
    # Get bottlenecks
    queue_depths = monitor.get_queue_depths()
    bottlenecks = bottleneck_detector.detect_bottlenecks(queue_depths)
    
    # Combine into alerts
    alerts = []
    for breach in breaches[:5]:  # Top 5 breach risks
        alerts.append({
            "type": "sla_breach_risk",
            "severity": "critical" if breach['breach_probability'] > 0.8 else "warning",
            "message": f"Case {breach['case_id']} at {breach['breach_probability']*100:.0f}% breach risk",
            "details": breach
        })
    
    for bn in bottlenecks:
        alerts.append({
            "type": "bottleneck",
            "severity": bn['severity'],
            "message": bn['recommendation'],
            "details": bn
        })
    
    return {"alerts": alerts}

@app.get("/state/wip")
def get_wip_state():
    """Get current Work-in-Progress state"""
    active_cases = monitor.get_active_cases()
    queue_depths = monitor.get_queue_depths()
    utilization = monitor.get_resource_utilization()
    
    update_queue_metrics(queue_depths)
    update_utilization_metrics(utilization)
    
    return {
        "active_case_count": len(active_cases['case_id'].unique()) if not active_cases.empty else 0,
        "queue_depths": queue_depths,
        "resource_utilization": utilization
    }
