-- Three-table order-to-cash schema used by the Operational Turing Test.
--
-- The schema declares only the *declarative* part of the operational rule
-- set Π:
--   * referential integrity (FK declarations)
--   * value-domain CHECKs (status enum, tier enum)
--   * the per-row quantity > 0 bound
--
-- Procedural rules (per-customer open-order limit, line/order total
-- derivation formulas, legal status-transition pairs) live in trigger.sql
-- and in the application code that drives generation.

CREATE TABLE customers (
    id           INTEGER PRIMARY KEY,
    country      VARCHAR(2),
    tier         VARCHAR(10) CHECK (tier IN ('bronze', 'silver', 'gold')),
    signup_date  DATE
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(id),
    status       VARCHAR(20)
                 CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    prev_status  VARCHAR(20),
    order_date   DATE,
    total        NUMERIC(10, 2)
);

CREATE TABLE order_items (
    id           INTEGER PRIMARY KEY,
    order_id     INTEGER REFERENCES orders(id),
    product_id   INTEGER,
    quantity     INTEGER CHECK (quantity > 0),
    unit_price   NUMERIC(10, 2),
    line_total   NUMERIC(10, 2)
);
