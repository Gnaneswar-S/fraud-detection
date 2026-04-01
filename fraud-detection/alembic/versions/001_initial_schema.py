"""Initial schema: transactions, predictions, audit_log

Revision ID: 001_initial
Revises:
Create Date: 2024-11-12 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- transactions ---
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.String(64), nullable=False),
        sa.Column("transaction_amt", sa.Float(), nullable=False),
        sa.Column("product_cd", sa.String(8), nullable=True),
        sa.Column("card_type", sa.String(16), nullable=True),
        sa.Column("p_emaildomain", sa.String(64), nullable=True),
        sa.Column("transaction_dt", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_transactions_id", "transactions", ["id"])
    op.create_index(
        "ix_transactions_transaction_id", "transactions", ["transaction_id"], unique=True
    )

    # --- predictions ---
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.String(64), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("fraud_label", sa.Boolean(), nullable=False),
        sa.Column("risk_tier", sa.String(16), nullable=False),
        sa.Column("top_features_json", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"])
    op.create_index(
        "ix_predictions_transaction_id", "predictions", ["transaction_id"]
    )

    # --- audit_log ---
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("endpoint", sa.String(128), nullable=True),
        sa.Column("username", sa.String(64), nullable=True),
        sa.Column("transaction_id", sa.String(64), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_id", "audit_log", ["id"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_id", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_predictions_transaction_id", table_name="predictions")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_transactions_transaction_id", table_name="transactions")
    op.drop_index("ix_transactions_id", table_name="transactions")
    op.drop_table("transactions")
