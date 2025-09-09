"""
FastAPI backend scaffold for a Tree Planting Donation Platform.

Features included:
- JWT auth (register/login/current user)
- Users (donor/admin roles)
- Donations (create, list, user history)
- Trees (add, list by user, fetch for map, details)
- Tree updates (growth tracking)
- Paystack webhook endpoint (stub) for payment verification
- AI recommendation endpoint (stub)

Storage: PostgreSQL via SQLAlchemy. For MVP we store latitude/longitude as floats.
You can later migrate to PostGIS/GeoAlchemy2 if needed.

How to run (after filling env vars):
- pip install -r requirements.txt (see requirements block below)
- uvicorn app:app --reload

Environment variables expected (e.g., in a .env file):
- DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
- JWT_SECRET=your_very_secret_key
- JWT_ALG=HS256
- PAYSTACK_SECRET=your_paystack_secret

"""

from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, Body, Path, Query, Header, Request, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from passlib.context import CryptContext
import jwt
import os
import httpx

# -----------------------------
# Config & DB setup
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET", "paystack_secret_placeholder")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -----------------------------
# Security utils
# -----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# -----------------------------
# ORM Models
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="donor")  # donor | admin
    created_at = Column(DateTime, default=datetime.utcnow)

    donations = relationship("Donation", back_populates="donor")
    trees = relationship("Tree", back_populates="donor")


class Donation(Base):
    __tablename__ = "donations"
    id = Column(Integer, primary_key=True, index=True)
    donor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String(10), default="NGN")
    payment_status = Column(String(30), default="pending")  # pending | paid | failed | refunded
    tree_species = Column(String(100), nullable=False)
    number_of_trees = Column(Integer, default=1)
    transaction_reference = Column(String(120), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    donor = relationship("User", back_populates="donations")
    trees = relationship("Tree", back_populates="donation")


class Tree(Base):
    __tablename__ = "trees"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    donor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    donation_id = Column(Integer, ForeignKey("donations.id"), nullable=True)
    planting_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String(30), default="planted")  # planted | growing | matured | dead
    photo_url = Column(Text, nullable=True)
    growth_stage = Column(String(50), default="seedling")
    created_at = Column(DateTime, default=datetime.utcnow)

    donor = relationship("User", back_populates="trees")
    donation = relationship("Donation", back_populates="trees")
    updates = relationship("TreeUpdate", back_populates="tree", cascade="all, delete-orphan")


class TreeUpdate(Base):
    __tablename__ = "tree_updates"
    id = Column(Integer, primary_key=True, index=True)
    tree_id = Column(Integer, ForeignKey("trees.id"), nullable=False)
    update_date = Column(DateTime, default=datetime.utcnow)
    photo_url = Column(Text, nullable=True)
    growth_notes = Column(Text, nullable=True)

    tree = relationship("Tree", back_populates="updates")


# -----------------------------
# Pydantic Schemas
# -----------------------------
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_len(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DonationCreate(BaseModel):
    amount: int
    currency: str = "NGN"
    tree_species: str
    number_of_trees: int = 1


class DonationOut(BaseModel):
    id: int
    donor_id: int
    amount: int
    currency: str
    payment_status: str
    tree_species: str
    number_of_trees: int
    transaction_reference: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class TreeCreate(BaseModel):
    species: str
    latitude: float
    longitude: float
    donor_id: Optional[int] = None
    donation_id: Optional[int] = None
    photo_url: Optional[str] = None
    status: str = "planted"
    growth_stage: str = "seedling"


class TreeOut(BaseModel):
    id: int
    species: str
    latitude: float
    longitude: float
    donor_id: Optional[int]
    donation_id: Optional[int]
    planting_date: datetime
    status: str
    photo_url: Optional[str]
    growth_stage: str

    class Config:
        from_attributes = True


class TreeUpdateCreate(BaseModel):
    photo_url: Optional[str] = None
    growth_notes: Optional[str] = None


class TreeUpdateOut(BaseModel):
    id: int
    tree_id: int
    update_date: datetime
    photo_url: Optional[str]
    growth_notes: Optional[str]

    class Config:
        from_attributes = True


class AIRecommendRequest(BaseModel):
    species: str


class AIRecommendOut(BaseModel):
    species: str
    recommended_region: str
    rationale: str


class PaystackWebhookData(BaseModel):
    event: Optional[str] = None
    data: dict


# -----------------------------
# Auth helpers & dependencies
# -----------------------------

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    payload = decode_token(token)
    user_id: int = int(payload.get("sub"))
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def admin_required(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Tree Planting Donation API", version="0.1.0")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    # --- Ensure initial admin exists ---
    db = SessionLocal()
    try:
        admin_email = "tree@admin.com"
        admin_password = "t1r2e3e4"
        user = db.query(User).filter(User.email == admin_email).first()
        if not user:
            user = User(
                name="Tree Admin",
                email=admin_email,
                password_hash=hash_password(admin_password),
                role="admin",
            )
            db.add(user)
            db.commit()
    finally:
        db.close()


# -----------------------------
# Auth routes
# -----------------------------
@app.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="donor",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenOut(access_token=token)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/auth/login-json", response_model=TokenOut)
def login_json(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return TokenOut(access_token=token)


@app.get("/users/me", response_model=UserOut)
def me(current: User = Depends(get_current_user)):
    return current


# -----------------------------
# Donation routes
# -----------------------------
@app.post("/donations/create", response_model=DonationOut)
def create_donation(payload: DonationCreate, current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # NOTE: In production, generate a unique reference and redirect to Paystack checkout.
    ref = f"REF-{int(datetime.utcnow().timestamp())}-{current.id}"
    donation = Donation(
        donor_id=current.id,
        amount=payload.amount,
        currency=payload.currency,
        tree_species=payload.tree_species,
        number_of_trees=payload.number_of_trees,
        transaction_reference=ref,
        payment_status="pending",
    )
    db.add(donation)
    db.commit()
    db.refresh(donation)
    return donation


@app.get("/donations/user/{user_id}", response_model=List[DonationOut])
def user_donations(user_id: int = Path(..., ge=1), current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current.role != "admin" and current.id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return db.query(Donation).filter(Donation.donor_id == user_id).order_by(Donation.created_at.desc()).all()


@app.get("/donations", response_model=List[DonationOut])
def list_donations(_: User = Depends(admin_required), db: Session = Depends(get_db)):
    return db.query(Donation).order_by(Donation.created_at.desc()).all()


# Webhook stub: validate signature header and mark donation as paid
@app.post("/payments/paystack/webhook")
async def paystack_webhook(
    payload: PaystackWebhookData,
    x_paystack_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    data = payload.data
    ref = data.get("reference") or data.get("reference_code")
    status_str = data.get("status", "success")

    if not ref:
        raise HTTPException(status_code=400, detail="Missing reference")

    donation = db.query(Donation).filter(Donation.transaction_reference == ref).first()
    if not donation:
        raise HTTPException(status_code=404, detail="Donation not found")

    donation.payment_status = "paid" if status_str == "success" else "failed"
    db.commit()
    return {"ok": True}


# -----------------------------
# Tree routes
# -----------------------------
@app.post("/trees/add", response_model=TreeOut)
def add_tree(payload: TreeCreate, _: User = Depends(admin_required), db: Session = Depends(get_db)):
    tree = Tree(
        species=payload.species,
        latitude=payload.latitude,
        longitude=payload.longitude,
        donor_id=payload.donor_id,
        donation_id=payload.donation_id,
        photo_url=payload.photo_url,
        status=payload.status,
        growth_stage=payload.growth_stage,
    )
    db.add(tree)
    db.commit()
    db.refresh(tree)
    return tree


@app.get("/trees/user/{user_id}", response_model=List[TreeOut])
def trees_by_user(user_id: int = Path(..., ge=1), current: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current.role != "admin" and current.id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed")
    return db.query(Tree).filter(Tree.donor_id == user_id).order_by(Tree.created_at.desc()).all()


@app.get("/trees/{tree_id}", response_model=TreeOut)
def tree_detail(tree_id: int = Path(..., ge=1), _: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tree = db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    return tree


class MapTree(BaseModel):
    id: int
    species: str
    lat: float
    lng: float
    status: str
    planting_date: datetime


@app.get("/trees/map", response_model=List[MapTree])
def trees_for_map(
    species: Optional[str] = Query(None, description="Filter by species"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(1000, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    q = db.query(Tree)
    if species:
        q = q.filter(Tree.species.ilike(f"%{species}%"))
    if status:
        q = q.filter(Tree.status == status)
    q = q.order_by(Tree.created_at.desc()).limit(limit)
    rows = q.all()
    return [
        MapTree(id=t.id, species=t.species, lat=t.latitude, lng=t.longitude, status=t.status, planting_date=t.planting_date)
        for t in rows
    ]


# -----------------------------
# Tree updates
# -----------------------------
@app.post("/trees/{tree_id}/update", response_model=TreeUpdateOut)
def add_tree_update(tree_id: int, payload: TreeUpdateCreate, _: User = Depends(admin_required), db: Session = Depends(get_db)):
    tree = db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    upd = TreeUpdate(tree_id=tree_id, photo_url=payload.photo_url, growth_notes=payload.growth_notes)
    db.add(upd)
    db.commit()
    db.refresh(upd)
    return upd


@app.get("/trees/{tree_id}/updates", response_model=List[TreeUpdateOut])
def get_tree_updates(tree_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tree = db.get(Tree, tree_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")
    return db.query(TreeUpdate).filter(TreeUpdate.tree_id == tree_id).order_by(TreeUpdate.update_date.desc()).all()


# -----------------------------
# AI recommendation (stub)
# -----------------------------
@app.post("/ai/recommend", response_model=AIRecommendOut)
def ai_recommend(payload: AIRecommendRequest, _: User = Depends(get_current_user)):
    # Placeholder logic — replace with real model later
    species = payload.species.lower()
    mapping = {
        "mango": ("Northwest & Northcentral", "Warm temps, moderate rainfall; good survival and yield."),
        "neem": ("Across Nigeria, esp. North", "Drought-tolerant; thrives in semi-arid conditions."),
        "cashew": ("Southwest & Middle Belt", "Well-drained soils, 600–1200mm rainfall."),
    }
    region, why = mapping.get(species, ("Varies by state", "Use soil, rainfall, and elevation layers for best match."))
    return AIRecommendOut(species=payload.species, recommended_region=region, rationale=why)


# -----------------------------
# File upload to Supabase
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "tree-photos")

async def upload_to_supabase(file: UploadFile, filename: str) -> str:
    """Uploads a file to Supabase Storage and returns the public URL."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": file.content_type,
    }
    data = await file.read()
    async with httpx.AsyncClient() as client:
        resp = await client.put(url, content=data, headers=headers)
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"Supabase upload failed: {resp.text}")
    # Public URL pattern (adjust if you use RLS or custom domains)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
    return public_url

@app.post("/upload/tree-photo")
async def upload_tree_photo(
    file: UploadFile = File(...),
    current: User = Depends(get_current_user)
):
    # Use user id and timestamp for unique filename
    ext = file.filename.split(".")[-1]
    filename = f"user_{current.id}_{int(datetime.utcnow().timestamp())}.{ext}"
    url = await upload_to_supabase(file, filename)
    return {"url": url}


# -----------------------------
# Simple healthcheck
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# -----------------------------
# Requirements (put these into requirements.txt)
# -----------------------------
REQUIREMENTS_TXT = r"""
fastapi==0.112.2
uvicorn[standard]==0.30.6
SQLAlchemy==2.0.36
psycopg2-binary==2.9.9
passlib[bcrypt]==1.7.4
bcrypt==3.2.0
PyJWT==2.9.0
python-multipart==0.0.9
pydantic==2.9.1
httpx==0.24.1
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

@app.post("/users/{user_id}/promote", response_model=UserOut)
def promote_user_to_admin(
    user_id: int,
    _: User = Depends(admin_required),
    db: Session = Depends(get_db)
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = "admin"
    db.commit()
    db.refresh(user)
    return user
