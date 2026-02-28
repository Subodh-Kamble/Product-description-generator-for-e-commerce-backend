import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# load envvars from a .env file if present (local development)
load_dotenv()

# DATABASE_URL should be set by Render in production.  Fallback to sqlite for
# local development so that the app can run without additional setup.
#
# Example PostgreSQL URL:
#   postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./products.db")

# If using sqlite we need the check_same_thread flag so that the connection can
# be used across threads (uvicorn workers).
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()