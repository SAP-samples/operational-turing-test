-- Status-transition rule, enforced procedurally.
--
-- The legal-transition set is a four-element relation over
-- (prev_status, status). It is *not* expressible as a CHECK constraint on
-- a single row (the rule is row-pair-temporal in spirit), and it is not
-- declared anywhere in schema.sql. A model with access only to the schema
-- text cannot reconstruct it.
--
-- This trigger is the procedural component of operational rule family (iv).

CREATE OR REPLACE FUNCTION check_status_transition()
RETURNS TRIGGER AS $$
DECLARE
    allowed CONSTANT TEXT[][] := ARRAY[
        ['pending',   'shipped'],
        ['shipped',   'delivered'],
        ['pending',   'cancelled'],
        ['pending',   'pending']     -- initial state
    ];
BEGIN
    IF NOT (ARRAY[NEW.prev_status, NEW.status] = ANY (allowed))
    THEN
        RAISE EXCEPTION 'illegal transition % -> %',
            NEW.prev_status, NEW.status;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER orders_status_transition
BEFORE INSERT OR UPDATE ON orders
FOR EACH ROW EXECUTE FUNCTION check_status_transition();
