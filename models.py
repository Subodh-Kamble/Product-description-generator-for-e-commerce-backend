from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship

from database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    specs = Column(Text, nullable=False)
    features = Column(Text, nullable=False)
    category = Column(String, nullable=False)

    descriptions = relationship("Description", back_populates="product", cascade="all, delete-orphan")


class Description(Base):
    __tablename__ = "descriptions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    description = Column(Text, nullable=False)
    tone = Column(String, nullable=False)
    language = Column(String, nullable=False)
    keywords = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # analysis fields
    quality_score = Column(Integer)
    seo_score = Column(Integer)
    engagement_score = Column(Integer)
    conversion_score = Column(Integer)
    overall_score = Column(Float)
    analysis_notes = Column(Text)
    analyzed_at = Column(DateTime)

    product = relationship("Product", back_populates="descriptions")