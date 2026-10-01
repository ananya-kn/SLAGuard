import json
import os
from kafka import KafkaProducer
from datetime import datetime

class EventProducer:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
        self.topic = "appian-events"
        self.producer = None

    def connect(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
            )
            print(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")

    def send_event(self, event: dict):
        if not self.producer:
            self.connect()
        
        if self.producer:
            self.producer.send(self.topic, event)
            self.producer.flush()
