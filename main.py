"""
Backend - FastAPI Application

Available Endpoints:
    GET /                    - Welcome message
    GET /health              - Health check endpoint
    GET /api/random-quote    - Sample endpoint to connect Frontend and Backend (generates random quote using Gemini LLM)

Database:
    - Locally sqlite:///./products.db when DATABASE_URL is unset
    - In production (Render), set DATABASE_URL to a PostgreSQL URI:
      postgresql://user:password@host:port/dbname

To run:
    uvicorn main:app --reload                 # dev
    uvicorn main:app --host 0.0.0.0 --port $PORT  # prod (Render)

Setup:
    1. pip install -r requirements.txt
    2. Create a .env file with necessary variables:
         GOOGLE_API_KEY=...
         DATABASE_URL=postgresql://user:pass@host:port/dbname
    3. Run uvicorn as above.
"""

import os
from typing import Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

from sqlalchemy.orm import Session
from sqlalchemy import or_
from database import SessionLocal, engine, Base
import models


# load environment variables from .env
load_dotenv()

# create tables (works for both sqlite & postgres)
Base.metadata.create_all(bind=engine)

# initialize Google AI – environment variable should contain your API key.
# we look for LLM_API_KEY for clarity; older documentation may refer to
# "GOOGLE_API_KEY" which is functionally equivalent.
api_key = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: LLM_API_KEY (or GOOGLE_API_KEY) not found in environment variables.")
    print("Please create a .env file with: LLM_API_KEY=your_api_key_here")
    genai_configured = False
else:
    genai.configure(api_key=api_key)
    genai_configured = True

app = FastAPI(title="Backend API", version="0.1.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    """FastAPI dependency that yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic schemas for request validation
class ProductCreate(BaseModel):
    user_id: str
    name: str
    specs: str
    features: str
    category: str


class ProductUpdate(BaseModel):
    name: str
    specs: str
    features: str


class DescriptionGenerateRequest(BaseModel):
    tone: str
    language: str
    num_variations: int = 3


# Database helper functions using SQLAlchemy ORM

def insert_product(db: Session, product: ProductCreate) -> int:
    db_product = models.Product(
        user_id=product.user_id,
        name=product.name,
        specs=product.specs,
        features=product.features,
        category=product.category,
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product.id


def get_product_by_id(db: Session, product_id: int) -> models.Product | None:
    return db.query(models.Product).filter(models.Product.id == product_id).first()


def get_description_by_id(db: Session, description_id: int) -> models.Description | None:
    return db.query(models.Description).filter(models.Description.id == description_id).first()


def save_description_analysis(
    db: Session,
    description_id: int,
    quality_score: int,
    seo_score: int,
    engagement_score: int,
    conversion_score: int,
    overall_score: float,
    analysis_notes: str,
) -> models.Description | None:
    desc = db.query(models.Description).filter(models.Description.id == description_id).first()
    if not desc:
        return None
    desc.quality_score = quality_score
    desc.seo_score = seo_score
    desc.engagement_score = engagement_score
    desc.conversion_score = conversion_score
    desc.overall_score = overall_score
    desc.analysis_notes = analysis_notes
    desc.analyzed_at = datetime.now(timezone.utc)
    db.commit()
    return desc


def analyze_description_with_llm(description_text: str) -> dict[str, Any]:
    response = analysis_llm.invoke(
        analysis_prompt.format(description=description_text)
    )
    raw = response.content.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found. Raw output: {raw}")
    json_str = raw[start:end]
    data = json.loads(json_str)
    quality = int(data["quality"])
    seo = int(data["seo"])
    engagement = int(data["engagement"])
    conversion = int(data["conversion"])
    notes = data.get("notes", "")
    overall = round((quality + seo + engagement + conversion) / 4, 2)
    return {
        "quality": quality,
        "seo": seo,
        "engagement": engagement,
        "conversion": conversion,
        "overall": overall,
        "notes": notes,
    }


def get_analyzed_descriptions_by_product(db: Session, product_id: int) -> list[dict[str, Any]]:
    descs = (
        db.query(models.Description)
        .filter(
            models.Description.product_id == product_id,
            models.Description.overall_score.isnot(None),
        )
        .order_by(models.Description.overall_score.desc())
        .all()
    )
    return [
        {
            "id": d.id,
            "quality_score": d.quality_score,
            "seo_score": d.seo_score,
            "engagement_score": d.engagement_score,
            "conversion_score": d.conversion_score,
            "overall_score": d.overall_score,
            "analysis_notes": d.analysis_notes,
        }
        for d in descs
    ]


def calculate_analysis_averages(descriptions: list[dict[str, Any]]) -> dict[str, float]:
    total: dict[str, float] = {"quality": 0.0, "seo": 0.0, "engagement": 0.0, "conversion": 0.0, "overall": 0.0}
    count = len(descriptions)
    for d in descriptions:
        total["quality"] += d["quality_score"]
        total["seo"] += d["seo_score"]
        total["engagement"] += d["engagement_score"]
        total["conversion"] += d["conversion_score"]
        total["overall"] += d["overall_score"]
    return {k: round(v / count, 2) for k, v in total.items()}


def get_products_by_user(db: Session, user_id: str) -> list[tuple[int, str, str]]:
    return (
        db.query(models.Product.id, models.Product.name, models.Product.category)
        .filter(models.Product.user_id == user_id)
        .order_by(models.Product.id.desc())
        .all()
    )


def search_products(db: Session, user_id: str, query: str, category: Optional[str]) -> list[tuple[int, str, str]]:
    q = (
        db.query(models.Product.id, models.Product.name, models.Product.category)
        .filter(models.Product.user_id == user_id)
        .filter(
            or_(
                models.Product.name.ilike(f"%{query}%"),
                models.Product.specs.ilike(f"%{query}%"),
                models.Product.features.ilike(f"%{query}%"),
            )
        )
    )
    if category:
        q = q.filter(models.Product.category == category)
    return q.order_by(models.Product.id.desc()).all()


def save_description(
    db: Session,
    product_id: int,
    description: str,
    tone: str,
    language: str,
    keywords: Optional[str] = None,
) -> models.Description:
    db_desc = models.Description(
        product_id=product_id,
        description=description,
        tone=tone,
        language=language,
        keywords=keywords,
    )
    db.add(db_desc)
    db.commit()
    db.refresh(db_desc)
    return db_desc


def get_product_with_descriptions(db: Session, product_id: int) -> dict[str, Any] | None:
    prod = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not prod:
        return None
    return {
        "product": {
            "id": prod.id,
            "user_id": prod.user_id,
            "name": prod.name,
            "specs": prod.specs,
            "features": prod.features,
            "category": prod.category,
        },
        "descriptions": [
            {
                "id": d.id,
                "description": d.description,
                "tone": d.tone,
                "language": d.language,
                "keywords": d.keywords,
                "created_at": d.created_at,
                "quality_score": d.quality_score,
                "seo_score": d.seo_score,
                "engagement_score": d.engagement_score,
                "conversion_score": d.conversion_score,
                "overall_score": d.overall_score,
                "analysis_notes": d.analysis_notes,
                "analyzed_at": d.analyzed_at,
            }
            for d in prod.descriptions
        ],
    }


def update_product(db: Session, product_id: int, name: str, specs: str, features: str) -> int:
    prod = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not prod:
        return 0
    prod.name = name
    prod.specs = specs
    prod.features = features
    db.commit()
    return 1


def delete_product_and_descriptions(db: Session, product_id: int, user_id: str) -> bool:
    prod = (
        db.query(models.Product)
        .filter(models.Product.id == product_id, models.Product.user_id == user_id)
        .first()
    )
    if not prod:
        return False
    db.delete(prod)
    db.commit()
    return True


def get_product_specs_for_keywords(db: Session, product_id: int) -> dict[str, str] | None:
    prod = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not prod:
        return None
    return {
        "name": prod.name,
        "specs": prod.specs,
        "features": prod.features,
        "category": prod.category,
    }


# supply API key explicitly so the class doesn’t rely on a specific
# environment variable name (it understands GOOGLE_API_KEY/GEMINI_API_KEY).
# our code prefers LLM_API_KEY for clarity, so forward that value here.
api_val = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_val,
)

description_prompt = PromptTemplate(
    input_variables=["name", "category", "specs", "features", "tone", "language"],
    template="""
You are an expert e-commerce copywriter.

Write a {tone}, SEO-optimized product description in {language}.

Product Name: {name}
Category: {category}
Specifications: {specs}
Features: {features}

Requirements:
- Naturally include SEO keywords
- Be engaging and conversion-focused
- Keep it concise and clear
"""
)

def generate_descriptions(
    product: dict[str, str],
    tone: str,
    language: str,
    num_variations: int,
) -> list[str]:
    results: list[str] = []

    for _ in range(num_variations):
        prompt = description_prompt.format(
            name=product["name"],
            category=product["category"],
            specs=product["specs"],
            features=product["features"],
            tone=tone,
            language=language
        )

        response = llm.invoke(prompt)
        results.append(response.content.strip())

    return results


keyword_prompt = PromptTemplate(
    input_variables=["name", "category", "specs", "features"],
    template="""
You are an SEO expert.

Extract 8–12 SEO keywords.
Return ONLY comma-separated keywords.

Rules:
- Keywords should be short phrases
- Relevant for search engines
- No explanations
- Return ONLY a comma-separated list

Product Name: {name}
Category: {category}
Specifications: {specs}
Features: {features}
"""
)

keyword_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    api_key=api_val,
)

analysis_prompt = PromptTemplate(
    input_variables=["description"],
    template="""
Return ONLY valid JSON.
No text.
No markdown.

Format exactly:
{{
  "quality": number,
  "seo": number,
  "engagement": number,
  "conversion": number,
  "notes": "text"
}}

Description:
{description}
"""
)

analysis_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    api_key=api_val,
)


def extract_seo_keywords(product_data: dict[str, str]) -> list[str]:
    prompt = keyword_prompt.format(**product_data)
    response = keyword_llm.invoke(prompt)
    return [k.strip() for k in response.content.split(",") if k.strip()]



@app.get("/")
async def root():
    return {"message": "Hello from AI Interviewer Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/random-quote")
async def get_random_quote():
    """
    Sample endpoint to connect Frontend and Backend.
    This is a simple example endpoint that generates a random inspirational quote using Google's Gemini LLM.
    Students can use this endpoint to practice connecting their React frontend to the FastAPI backend.
    
    Returns:
        JSON response with AI-generated random quote
    """
    if not genai_configured:
        raise HTTPException(
            status_code=500,
            detail="Google API key not configured. Please set GOOGLE_API_KEY in your .env file."
        )
    
    try:
        # Make a simple LLM call to generate a random quote
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Tell me a random inspirational quote")
        
        return {
            "success": True,
            "message": "Random quote generated successfully",
            "data": {
                "quote": response.text,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating quote: {str(e)}"
        )


@app.post("/api/products")
async def create_product(product: ProductCreate, db: Session = Depends(get_db)):

    # Extra safety validation
    if not all([
        product.user_id,
        product.name,
        product.specs,
        product.features,
        product.category
    ]):
        raise HTTPException(
            status_code=400,
            detail="All fields are required"
        )

    try:
        product_id = insert_product(db, product)

        return {
            "success": True,
            "message": "Product created successfully",
            "data": {
                "id": product_id,
                "user_id": product.user_id,
                "name": product.name,
                "specs": product.specs,
                "features": product.features,
                "category": product.category,
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating product: {str(e)}"
        )

@app.get("/api/products")
async def get_user_products(user_id: str, db: Session = Depends(get_db)):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    try:
        products = get_products_by_user(db, user_id)

        return {
            "success": True,
            "count": len(products),
            "data": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "category": p["category"]
                    
                }
                for p in products
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/search")
async def search_products_api(
    user_id: str,
    query: str,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required")

    try:
        results = search_products(db, user_id, query, category)

        return {
            "success": True,
            "count": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/api/products/{product_id}")
async def get_product_detail(product_id: int, db: Session = Depends(get_db)):
    try:
        data = get_product_with_descriptions(db, product_id)

        if not data:
            raise HTTPException(status_code=404, detail="Product not found")

        return {
            "success": True,
            "data": data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching product details: {str(e)}"
        )

@app.patch("/api/products/{product_id}")
async def update_product_api(product_id: int, product: ProductUpdate, db: Session = Depends(get_db)):
    existing = get_product_by_id(db, product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")

    rows = update_product(
        db,
        product_id,
        product.name,
        product.specs,
        product.features
    )

    if rows == 0:
        raise HTTPException(status_code=400, detail="Update failed")

    return {
        "success": True,
        "message": "Product updated successfully",
        "data": {
            "id": product_id,
            "name": product.name,
            "specs": product.specs,
            "features": product.features
        }
    }

@app.delete("/api/products/{product_id}")
async def delete_product(product_id: int, user_id: str, db: Session = Depends(get_db)):
    try:
        deleted = delete_product_and_descriptions(db, product_id, user_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Product not found or unauthorized"
            )

        return {
            "success": True,
            "message": "Product deleted successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete product: {str(e)}"
        )

@app.post("/api/products/{product_id}/generate")
async def generate_product_descriptions(
    product_id: int,
    request: DescriptionGenerateRequest,
    db: Session = Depends(get_db),
):
    product_obj = get_product_by_id(db, product_id)

    if not product_obj:
        raise HTTPException(status_code=404, detail="Product not found")

    product_data = {
        "name": product_obj.name,
        "category": product_obj.category,
        "specs": product_obj.specs,
        "features": product_obj.features,
    }

    try:
        variations = min(request.num_variations, 5)

        descriptions = generate_descriptions(
            product=product_data,
            tone=request.tone,
            language=request.language,
            num_variations=variations
        )

        response_data = []

        for desc in descriptions:
            save_description(
                db,
                product_id=product_id,
                description=desc,
                tone=request.tone,
                language=request.language,
                keywords="SEO keywords auto-generated"
            )

            response_data.append({
                "description": desc,
                "tone": request.tone,
                "language": request.language
            })

        return {
            "success": True,
            "count": len(response_data),
            "descriptions": response_data
        }

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            raise HTTPException(
                status_code=429,
                detail="AI quota exhausted. Please try again later."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Description generation failed: {str(e)}"
        )

@app.post("/api/products/{product_id}/descriptions/{desc_id}/analyze")
async def analyze_description_endpoint(product_id: int, desc_id: int, db: Session = Depends(get_db)):
    try:
        # 1. Fetch description
        description_obj = get_description_by_id(db, desc_id)

        if not description_obj:
            raise HTTPException(
                status_code=404,
                detail="Description not found"
            )

        # Optional safety check (description belongs to product)
        if description_obj.product_id != product_id:
            raise HTTPException(
                status_code=400,
                detail="Description does not belong to this product"
            )

        # 2. Analyze using LLM
        analysis = analyze_description_with_llm(
            description_obj.description
        )

        # 3. Save analysis results
        save_description_analysis(
            db,
            description_id=desc_id,
            quality_score=analysis["quality"],
            seo_score=analysis["seo"],
            engagement_score=analysis["engagement"],
            conversion_score=analysis["conversion"],
            overall_score=analysis["overall"],
            analysis_notes=analysis["notes"],
        )

        # 4. Return analysis
        return {
            "success": True,
            "description_id": desc_id,
            "analysis": analysis
        }

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=500,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Description analysis failed: {str(e)}"
        )

@app.post("/api/products/{product_id}/compare-descriptions")
async def compare_descriptions(product_id: int, db: Session = Depends(get_db)):
    try:
        analyzed = get_analyzed_descriptions_by_product(db, product_id)

        if not analyzed:
            raise HTTPException(
                status_code=400,
                detail="No analyzed descriptions available for comparison"
            )

        best = analyzed[0]

        return {
            "success": True,
            "count": len(analyzed),
            "best_description_id": best["id"],
            "rankings": analyzed
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )

@app.get("/api/products/{product_id}/analytics")
async def get_product_analytics(product_id: int, db: Session = Depends(get_db)):
    try:
        analyzed = get_analyzed_descriptions_by_product(db, product_id)

        if not analyzed:
            raise HTTPException(
                status_code=404,
                detail="No analysis data available for this product"
            )

        averages = calculate_analysis_averages(analyzed)
        best = analyzed[0]

        return {
            "success": True,
            "count": len(analyzed),
            "best_description_id": best["id"],
            "averages": averages,
            "rankings": analyzed
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analytics fetch failed: {str(e)}"
        )


@app.post("/api/products/{product_id}/keywords")
async def extract_product_keywords(product_id: int, db: Session = Depends(get_db)):
    product = get_product_specs_for_keywords(db, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    keywords = extract_seo_keywords(product)
    return {"success": True, "keywords": keywords}

