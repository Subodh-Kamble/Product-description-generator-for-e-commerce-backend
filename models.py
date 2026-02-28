from datetime import datetime, timezone
from sqlalchemy import Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    specs: Mapped[str] = mapped_column(Text, nullable=False)
    features: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)

    descriptions: Mapped[list["Description"]] = relationship(
        "Description", back_populates="product", cascade="all, delete-orphan"
    )


class Description(Base):
    __tablename__ = "descriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    tone: Mapped[str] = mapped_column(String, nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False)
    keywords: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # analysis fields
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seo_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    engagement_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conversion_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    product: Mapped["Product"] = relationship("Product", back_populates="descriptions")