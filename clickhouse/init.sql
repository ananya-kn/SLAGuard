CREATE TABLE IF NOT EXISTS events (
    event_id UInt64,
    case_id String,
    activity String,
    status String,
    timestamp DateTime64(3),
    resource String,
    sla_deadline DateTime64(3),
    priority UInt8,
    -- c-column format aliases for Appian compatibility
    c0 String ALIAS activity,
    c1 String ALIAS status,
    c2 DateTime64(3) ALIAS timestamp,
    c3 DateTime64(3) ALIAS sla_deadline,
    c4 UInt8 ALIAS priority,
    c5 String ALIAS case_id
) ENGINE = MergeTree()
ORDER BY (case_id, timestamp);
