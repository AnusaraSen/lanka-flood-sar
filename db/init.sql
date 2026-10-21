-- Draft schema for experiment and evaluation results (read by Grafana).
-- Starting point only: it will be refined in Week 2 when result logging is built.

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id        SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    name          TEXT NOT NULL,
    model         TEXT NOT NULL,        -- e.g. otsu, rf, unet_resnet34
    seed          INTEGER,
    config_json   JSONB,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id            SERIAL PRIMARY KEY,
    run_id        INTEGER REFERENCES experiment_runs (run_id) ON DELETE CASCADE,
    event         TEXT NOT NULL,        -- flood event the score is computed on
    split         TEXT NOT NULL,        -- train, val, test, held_out
    metric        TEXT NOT NULL,        -- iou, f1, precision, recall, ece ...
    value         DOUBLE PRECISION NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_curves (
    id            SERIAL PRIMARY KEY,
    run_id        INTEGER REFERENCES experiment_runs (run_id) ON DELETE CASCADE,
    epoch         INTEGER NOT NULL,
    metric        TEXT NOT NULL,        -- train_loss, val_loss, val_iou ...
    value         DOUBLE PRECISION NOT NULL
);
