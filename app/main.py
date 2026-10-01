from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.generator import DataGenerator
from app.kafka_producer import EventProducer
from app.schemas import GenerationRequest
from datetime import datetime

app = FastAPI(title="Appian Data Simulator")

# Enable CORS for demo UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = DataGenerator()
producer = EventProducer()

@app.on_event("startup")
async def startup_event():
    producer.connect()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/generate")
async def generate_data(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate events and send to Kafka in background"""
    start_date = request.start_date or datetime.now()
    
    cases = generator.generate_cases(request.num_cases, start_date)
    events = generator.generate_events(cases)
    
    # Send to Kafka in background
    background_tasks.add_task(send_events_to_kafka, events)
    
    return {
        "message": f"Generating {len(events)} events for {request.num_cases} cases",
        "case_count": len(cases),
        "event_count": len(events)
    }

def send_events_to_kafka(events):
    for event in events:
        producer.send_event(event)
    print(f"Sent {len(events)} events to Kafka")
