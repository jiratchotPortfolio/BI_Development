-- Schema Design for Customer Support Intelligence
-- Optimized for high-volume read/write operations

CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id SERIAL PRIMARY KEY,
    agoda_vip_status VARCHAR(50),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer_profiles(customer_id),
    issue_category VARCHAR(100) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('OPEN', 'RESOLVED', 'PENDING')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_logs (
    log_id SERIAL PRIMARY KEY,
    ticket_id INT REFERENCES support_tickets(ticket_id),
    message_content TEXT NOT NULL,
    sender_type VARCHAR(10) CHECK (sender_type IN ('USER', 'AGENT', 'BOT')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating indexes to optimize retrieval speed for the RAG engine
CREATE INDEX idx_ticket_customer ON support_tickets(customer_id);
CREATE INDEX idx_logs_ticket ON conversation_logs(ticket_id);
CREATE INDEX idx_logs_content_gin ON conversation_logs USING gin(to_tsvector('english', message_content));
