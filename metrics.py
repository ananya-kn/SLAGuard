from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics
sla_breach_probability = Gauge(
    'sla_breach_probability', 
    'Probability of SLA breach for a case',
    ['case_id', 'tier']
)

queue_depth = Gauge(
    'queue_depth',
    'Current queue depth per activity',
    ['activity']
)

resource_utilization = Gauge(
    'resource_utilization',
    'Resource pool utilization',
    ['pool']
)

forecast_requests = Counter(
    'forecast_requests_total',
    'Total forecast requests'
)

forecast_latency = Histogram(
    'forecast_latency_seconds',
    'Forecast request latency'
)

def update_queue_metrics(depths: dict):
    """Update queue depth metrics"""
    for activity, depth in depths.items():
        queue_depth.labels(activity=activity).set(depth)

def update_utilization_metrics(utilization: dict):
    """Update resource utilization metrics"""
    for resource, util in utilization.items():
        resource_utilization.labels(pool=resource).set(util)

def get_metrics():
    """Return Prometheus metrics"""
    return generate_latest()

def get_content_type():
    return CONTENT_TYPE_LATEST
