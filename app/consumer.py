import os
import json
import threading
from kafka import KafkaConsumer
from clickhouse_driver import Client
from datetime import datetime
from typing import Dict, Any

class EventConsumer:
    """Consumes events from Kafka and writes to ClickHouse"""
    
    def __init__(self):
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
        self.ch_host = os.getenv("CLICKHOUSE_HOST", "localhost")
        self.topic = "appian-events"
        self.consumer = None
        self.ch_client = None
        self.running = False
        self._thread = None
    
    def connect(self):
        """Connect to Kafka and ClickHouse"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.kafka_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='sla-monitor-consumer',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            self.ch_client = Client(host=self.ch_host)
            print(f"EventConsumer connected to Kafka at {self.kafka_servers} and ClickHouse at {self.ch_host}")
            return True
        except Exception as e:
            print(f"EventConsumer failed to connect: {e}")
            return False
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string to datetime"""
        if isinstance(ts_str, datetime):
            return ts_str
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    
    def _insert_event(self, event: Dict[str, Any]):
        """Insert a single event into ClickHouse"""
        try:
            query = """
            INSERT INTO events (event_id, case_id, activity, status, timestamp, resource, sla_deadline, priority)
            VALUES
            """
            data = [(
                event.get('event_id', 0),
                event.get('case_id', ''),
                event.get('activity', ''),
                event.get('status', ''),
                self._parse_timestamp(event.get('timestamp', datetime.now())),
                event.get('resource', ''),
                self._parse_timestamp(event.get('sla_deadline', datetime.now())),
                event.get('priority', 3)
            )]
            self.ch_client.execute(query, data)
        except Exception as e:
            print(f"Failed to insert event: {e}")
    
    def _consume_loop(self):
        """Main consumer loop"""
        print("EventConsumer: Starting consume loop...")
        batch = []
        batch_size = 100
        
        while self.running:
            try:
                # Poll with timeout
                messages = self.consumer.poll(timeout_ms=1000)
                for topic_partition, records in messages.items():
                    for record in records:
                        self._insert_event(record.value)
                        
            except Exception as e:
                print(f"EventConsumer error: {e}")
    
    def start(self):
        """Start consuming in background thread"""
        if not self.consumer:
            if not self.connect():
                return False
        
        self.running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        print("EventConsumer: Background consumer started")
        return True
    
    def stop(self):
        """Stop the consumer"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.consumer:
            self.consumer.close()
