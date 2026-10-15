from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Textbook(Base):
    __tablename__ = "textbooks"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    file_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    status: Mapped[str] = mapped_column(
        String(30), default="processed", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    chunks = relationship(
        "TextbookChunk",
        back_populates="textbook",
        cascade="all, delete-orphan",
    )


class TextbookChunk(Base):
    __tablename__ = "textbook_chunks"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    textbook_id: Mapped[int] = mapped_column(
        ForeignKey("textbooks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    text: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    embedding = mapped_column(Vector(384), nullable=False)

    textbook = relationship(
        "Textbook",
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint(
            "textbook_id",
            "chunk_index",
            name="uq_textbook_chunk_index",
        ),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    student_name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    student_usn: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    obtained_marks: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    total_marks: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    percentage: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    details: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class AnswerScript(Base):
    __tablename__ = "answer_scripts"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    file_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )

    student_name: Mapped[str] = mapped_column(
        String(255), nullable=True
    )

    student_usn: Mapped[str] = mapped_column(
        String(100), nullable=True, index=True
    )

    extracted_text: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    parsed_questions: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    avg_confidence: Mapped[float] = mapped_column(
        Float, default=0.0
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

