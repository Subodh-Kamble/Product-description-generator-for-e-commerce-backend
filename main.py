"""
Backend - FastAPI Application

Available Endpoints:
    GET /                    - Welcome message
    GET /health              - Health check endpoint
    GET /api/random-quote    - Sample endpoint to connect Frontend and Backend (generates random quote using Gemini LLM)

To run this server:
    uvicorn main:app --reload

The server will start at: http://localhost:8000
API documentation will be available at: http://localhost:8000/docs

Setup:
    1. Install dependencies: pip install -r requirements.txt
    2. Get your Google API key from: https://makersuite.google.com/app/apikey
    3. Create a .env file in the Backend directory with: GOOGLE_API_KEY=your_api_key_here
    4. Run the server: uvicorn main:app --reload
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

# Load environment variables from .env file
load_dotenv()

# Initialize API key
api_key = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: LLM_API_KEY (or GOOGLE_API_KEY) not found in environment variables.")
    print("Please create a .env file with: LLM_API_KEY=your_api_key_here")
    genai_configured = False
else:
    # api_key is passed directly to ChatGoogleGenerativeAI instances
    genai_configured = True

# Initialize LLM clients
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key,
)

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # Startup
    create_products_table()
    create_descriptions_table()
    migrate_descriptions_analysis_columns()
    yield
    # Shutdown (if needed)

app = FastAPI(title="Backend API", version="0.1.0", lifespan=lifespan)
# Enable CORS (Cross-Origin Resource Sharing) to allow frontend to connect
# This is necessary because the frontend runs on a different port than the backend
# Without CORS, browsers will block requests from frontend to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000",
        "https://productgenai.netlify.app",
        "https://aiproductdescription.netlify.app"
    ],  # Vite default port and common React port
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Database location can be configured via environment variables on Render.
# For SQLite deployments you can set DB_PATH to a writable path like
# "/scratch/data/products.db" or use the default "/tmp/products.db".
# If you attach a managed database you could also set DATABASE_URL and
# switch to a different driver later.
DB_PATH = os.getenv("DB_PATH", "/tmp/products.db")

def get_db():  # type: ignore[no-untyped-def]
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_products_table():  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            specs TEXT NOT NULL,
            features TEXT NOT NULL,
            category TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def create_descriptions_table():  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            description TEXT NOT NULL,
            tone TEXT NOT NULL,
            language TEXT NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    conn.commit()
    conn.close()

def migrate_descriptions_analysis_columns():  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()

    columns = [
        ("quality_score", "INTEGER"),
        ("seo_score", "INTEGER"),
        ("engagement_score", "INTEGER"),
        ("conversion_score", "INTEGER"),
        ("overall_score", "REAL"),
        ("analysis_notes", "TEXT"),
        ("analyzed_at", "TIMESTAMP"),
    ]

    # Get existing columns
    cursor.execute("PRAGMA table_info(descriptions)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    # Add missing columns safely
    for column_name, column_type in columns:
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE descriptions ADD COLUMN {column_name} {column_type}"
            )

    conn.commit()
    conn.close()


# create_products_table()
# create_descriptions_table()
# migrate_descriptions_analysis_columns()


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

def insert_product(product: ProductCreate):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO products (user_id, name, specs, features, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            product.user_id,
            product.name,
            product.specs,
            product.features,
            product.category,
        )
    )

    conn.commit()
    product_id = cursor.lastrowid
    conn.close()

    return product_id

def get_product_by_id(product_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()
    conn.close()
    return product

def get_description_by_id(description_id: int):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM descriptions WHERE id = ?",
        (description_id,)
    )
    description = cursor.fetchone()

    conn.close()
    return description

def save_description_analysis(
    description_id: int,
    quality_score: int,
    seo_score: int,
    engagement_score: int,
    conversion_score: int,
    overall_score: float,
    analysis_notes: str
):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE descriptions
        SET quality_score = ?,
            seo_score = ?,
            engagement_score = ?,
            conversion_score = ?,
            overall_score = ?,
            analysis_notes = ?,
            analyzed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            quality_score,
            seo_score,
            engagement_score,
            conversion_score,
            overall_score,
            analysis_notes,
            description_id,
        )
    )

    conn.commit()
    conn.close()


def analyze_description_with_llm(description_text: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    response = analysis_llm.invoke(
        analysis_prompt.format(description=description_text)
    )

    # response.content can be str or list; convert to string
    raw = str(response.content).strip()  # type: ignore[arg-type]

    # extract JSON only
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

    return {  # type: ignore[return-value]
        "quality": quality,
        "seo": seo,
        "engagement": engagement,
        "conversion": conversion,
        "overall": overall,
        "notes": notes
    }

def get_analyzed_descriptions_by_product(product_id: int) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id,
               quality_score,
               seo_score,
               engagement_score,
               conversion_score,
               overall_score,
               analysis_notes
        FROM descriptions
        WHERE product_id = ?
          AND overall_score IS NOT NULL
        ORDER BY overall_score DESC
        """,
        (product_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def calculate_analysis_averages(descriptions: list[dict[str, Any]]) -> dict[str, float]:  # type: ignore[no-untyped-def]
    total: dict[str, float] = {
        "quality": 0.0,
        "seo": 0.0,
        "engagement": 0.0,
        "conversion": 0.0,
        "overall": 0.0,
    }

    count = len(descriptions)

    for d in descriptions:
        total["quality"] += d["quality_score"]
        total["seo"] += d["seo_score"]
        total["engagement"] += d["engagement_score"]
        total["conversion"] += d["conversion_score"]
        total["overall"] += d["overall_score"]

    return {
        k: round(v / count, 2) for k, v in total.items()
    }



def get_products_by_user(user_id: str) -> list[Any]:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, category
        FROM products
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )
    products = cursor.fetchall()
    conn.close()
    return products

def search_products(user_id: str, query: str, category: Optional[str]) -> list[Any]:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()

    sql = """
        SELECT id, name, category
        FROM products
        WHERE user_id = ?
          AND (
            name LIKE ?
            OR specs LIKE ?
            OR features LIKE ?
          )
    """

    params: list[Any] = [  # type: ignore[assignment]
        user_id,
        f"%{query}%",
        f"%{query}%",
        f"%{query}%"
    ]

    if category:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY id DESC"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def save_description(product_id: int, description: str, tone: str, language: str, keywords: Optional[str]) -> None:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO descriptions (product_id, description, tone, language, keywords)
        VALUES (?, ?, ?, ?, ?)
        """,
        (product_id, description, tone, language, keywords)
    )
    conn.commit()
    conn.close()

def get_product_with_descriptions(product_id: int) -> dict[str, Any] | None:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()

    # Get product
    cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()

    if not product:
        conn.close()
        return None

    # Get descriptions
    cursor.execute(
        "SELECT * FROM descriptions WHERE product_id = ? ORDER BY created_at DESC",
        (product_id,)
    )
    descriptions = cursor.fetchall()

    conn.close()

    return {  # type: ignore[return-value]
        "product": dict(product),
        "descriptions": [dict(d) for d in descriptions]
    }

def update_product(product_id: int, name: str, specs: str, features: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE products
        SET name = ?, specs = ?, features = ?
        WHERE id = ?
        """,
        (name, specs, features, product_id)
    )
    conn.commit()
    rows = cursor.rowcount
    conn.close()
    return rows

def delete_product_and_descriptions(product_id: int, user_id: str):
    conn = get_db()
    cursor = conn.cursor()

    # Verify product belongs to user
    cursor.execute(
        "SELECT id FROM products WHERE id = ? AND user_id = ?",
        (product_id, user_id)
    )
    product = cursor.fetchone()

    if not product:
        conn.close()
        return False

    # Delete descriptions first
    cursor.execute(
        "DELETE FROM descriptions WHERE product_id = ?",
        (product_id,)
    )

    # Delete product
    cursor.execute(
        "DELETE FROM products WHERE id = ?",
        (product_id,)
    )

    conn.commit()
    conn.close()
    return True

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    google_api_key=os.getenv("LLM_API_KEY")
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

def generate_descriptions(product: dict[str, str], tone: str, language: str, num_variations: int) -> list[str]:  # type: ignore[no-untyped-def]
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
        # response.content can be str or list; convert to string
        results.append(str(response.content).strip())  # type: ignore[arg-type]

    return results

def get_product_specs_for_keywords(product_id: int) -> dict[str, str] | None:  # type: ignore[no-untyped-def]
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, specs, features, category
        FROM products
        WHERE id = ?
    """, (product_id,))
    product = cursor.fetchone()
    conn.close()

    if not product:
        return None

    return {  # type: ignore[return-value]
        "name": product["name"],
        "specs": product["specs"],
        "features": product["features"],
        "category": product["category"],
    }

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
    api_key=api_key,
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
    api_key=api_key,
)


def extract_seo_keywords(product_data: dict[str, str]) -> list[str]:  # type: ignore[no-untyped-def]
    prompt = keyword_prompt.format(**product_data)
    response = keyword_llm.invoke(prompt)
    # response.content can be str or list; convert to string
    return [k.strip() for k in str(response.content).split(",") if k.strip()]  # type: ignore[arg-type]

@app.post("/api/products/{product_id}/keywords")
async def extract_product_keywords(product_id: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        product = get_product_specs_for_keywords(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        keywords = extract_seo_keywords(product)
        return {  # type: ignore[return-value]
            "success": True,
            "count": len(keywords),
            "keywords": keywords
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction failed: {str(e)}"
        )

@app.get("/")
async def root() -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"message": "Hello from AI Interviewer Backend!"}

@app.get("/health")
async def health_check() -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"status": "healthy"}

@app.get("/api/random-quote")
async def get_random_quote() -> dict[str, Any]:  # type: ignore[no-untyped-def]
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
        # Use ChatGoogleGenerativeAI to generate a random quote
        quote_llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=api_key,
        )
        response = quote_llm.invoke("Tell me a random inspirational quote")
        quote_text = str(response.content).strip()  # type: ignore[arg-type]
        
        return {  # type: ignore[return-value]
            "success": True,
            "message": "Random quote generated successfully",
            "data": {
                "quote": quote_text,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating quote: {str(e)}"
        )


@app.post("/api/products")
async def create_product(product: ProductCreate) -> dict[str, Any]:  # type: ignore[no-untyped-def]

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
        product_id = insert_product(product)

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
async def get_user_products(user_id: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    try:
        products = get_products_by_user(user_id)

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
    category: Optional[str] = None
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required")

    try:
        results = search_products(user_id, query, category)

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
async def get_product_detail(product_id: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        data = get_product_with_descriptions(product_id)

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
async def update_product_api(product_id: int, product: ProductUpdate) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    existing = get_product_by_id(product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")

    rows = update_product(
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
async def delete_product(product_id: int, user_id: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        deleted = delete_product_and_descriptions(product_id, user_id)

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
    request: DescriptionGenerateRequest
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    product = get_product_by_id(product_id)

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        variations = min(request.num_variations, 5)

        descriptions = generate_descriptions(
            product=product,
            tone=request.tone,
            language=request.language,
            num_variations=variations
        )

        response_data: list[dict[str, str]] = []

        for desc in descriptions:
            save_description(
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
async def analyze_description_endpoint(product_id: int, desc_id: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        # 1. Fetch description
        description = get_description_by_id(desc_id)

        if not description:
            raise HTTPException(
                status_code=404,
                detail="Description not found"
            )

        # Optional safety check (description belongs to product)
        if description["product_id"] != product_id:
            raise HTTPException(
                status_code=400,
                detail="Description does not belong to this product"
            )

        # 2. Analyze using LLM
        analysis = analyze_description_with_llm(
            description["description"]
        )

        # 3. Save analysis results
        save_description_analysis(
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
async def compare_descriptions(product_id: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        analyzed = get_analyzed_descriptions_by_product(product_id)

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
async def get_product_analytics(product_id: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    try:
        analyzed = get_analyzed_descriptions_by_product(product_id)

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

