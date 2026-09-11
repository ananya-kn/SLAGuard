#!/bin/bash
# Wait for Kafka to be ready
sleep 10
kafka-topics --create --if-not-exists --bootstrap-server kafka:9093 --replication-factor 1 --partitions 1 --topic appian-events
