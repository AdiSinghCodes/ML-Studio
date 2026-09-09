from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.datasets import router as dataset_router

from app.api.preprocessing import (
    router as preprocessing_router
)

from app.api.training import (
    router as training_router
)

from app.api.explain import (
    router as explain_router
)

from app.api.pipeline import (
    router as pipeline_router
)

from app.api.eda import (
    router as eda_router
)

from app.api.agent import (
    router as agent_router
)


app = FastAPI(
    title="ModelScope",
    description=(
        "Advanced Machine Learning and "
        "Explainable Machine Learning Platform"
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dataset APIs
app.include_router(
    dataset_router,
    prefix="/api"
)


# Preprocessing APIs
app.include_router(
    preprocessing_router,
    prefix="/api"
)


# Training and Hyperparameter APIs
app.include_router(
    training_router,
    prefix="/api"
)


# Explainable AI APIs
app.include_router(
    explain_router,
    prefix="/api"
)


# Automated Pipeline API
app.include_router(
    pipeline_router,
    prefix="/api"
)


# EDA API
app.include_router(
    eda_router,
    prefix="/api"
)


# AI Agent Recommendation API
app.include_router(
    agent_router,
    prefix="/api"
)


@app.get("/")
def root():
    return {
        "message": "ModelScope Backend is Running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
