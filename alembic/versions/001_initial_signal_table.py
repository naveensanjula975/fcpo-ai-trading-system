"""Initial signal table

Revision ID: 001
Revises: 
Create Date: 2025-11-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'signals',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('ticker', sa.String(length=10), nullable=False),
        sa.Column('signal_type', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.DECIMAL(precision=10, scale=2), nullable=False),
        sa.Column('tp1', sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('tp2', sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('tp3', sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('sl', sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('confidence', sa.DECIMAL(precision=5, scale=2), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('result', sa.DECIMAL(precision=10, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_signals_ticker'), 'signals', ['ticker'], unique=False)
    op.create_index(op.f('ix_signals_status'), 'signals', ['status'], unique=False)
    op.create_index(op.f('ix_signals_created_at'), 'signals', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_signals_created_at'), table_name='signals')
    op.drop_index(op.f('ix_signals_status'), table_name='signals')
    op.drop_index(op.f('ix_signals_ticker'), table_name='signals')
    op.drop_table('signals')
