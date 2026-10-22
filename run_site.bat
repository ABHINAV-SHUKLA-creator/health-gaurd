@echo off
rem Start HealthGuard (API + website) at http://127.0.0.1:8000
call .venv\Scripts\activate
cd backend
uvicorn app.main:app --reload
