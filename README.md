# RAMA OMD Analytics — Technical & User Guide

A modern analytics dashboard for marketing insights across banking and influencer campaigns. It includes authentication, CSV uploads, anomaly alerts, exports, and interactive visualizations (Chart.js + D3).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Stack](#stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start (Local)](#quick-start-local)
- [Default Accounts](#default-accounts)
- [How to Use the App](#how-to-use-the-app)
- [CSV Upload Format](#csv-upload-format)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Exporting Data](#exporting-data)
- [Troubleshooting](#troubleshooting)
- [Security Checklist](#security-checklist)
- [Publish to GitHub (Step-by-step)](#publish-to-github-step-by-step)
- [Hand-off Checklist](#hand-off-checklist)

---

## Project Overview
**RAMA OMD Analytics** provides a clean, trustworthy dashboard for revenue and item trends. The frontend is React (Vite/CRA compatible) with Chart.js and D3; the backend is a lightweight Python **Flask** API exposing `/api/login`, `/api/data`, `/api/upload`, and `/api/export`.

- **Authentication:** Admin & Viewer roles.
- **Uploads:** CSV with flexible column mapping.
- **Visualizations:** Pie, Bar, Line, and **Treemap** (D3) for clear revenue comparison.
- **Anomaly Alerts:** Outliers flagged after each upload.
- **Export:** Monthly summary CSV download.

> **Default API base**: `http://localhost:5001` (see [Configuration](#configuration)).

---

## Stack
- **Frontend:** React, Tailwind (CDN), Chart.js, D3, React Router
- **Backend:** Python 3 + Flask (with simple token-based login)
- **Build/Tooling:** Node.js 18+, npm, Git
- **Optional:** Docker for deployment

---

## Repository Structure
```
/ (repo root)
├─ frontend/                 # React app (src/App.js contains API_BASE)
│  ├─ public/
│  └─ src/
│     └─ App.js
├─ backend/                  # Flask API
│  ├─ app.py
│  ├─ requirements.txt
│  └─ (users.json or env vars for creds)
├─ README.md                 # This document
└─ .gitignore
```

> Your actual folders may differ; keep API base in sync with backend URL.

---

## Prerequisites
- **Node.js** 18+ and **npm**
- **Python** 3.9+ and **pip**
- **Git**
- (Optional) **virtualenv**

---

## Quick Start (Local)

### 1) Backend (Flask)
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Run the API on port 5001
python app.py
```
The API should be available at **http://localhost:5001**.

### 2) Frontend (React)
```bash
cd frontend
npm install
npm start   # or: npm run dev (if using Vite)
```
The UI runs on **http://localhost:3000** and talks to the backend on **http://localhost:5001**.

---

## Default Accounts

If you kept the sample backend users:
- **Admin:** `admin` / `admin`
- **Viewer:** `viewer` / `viewer`

> **Change immediately** for production. The backend can load users from a `users.json` or environment variables depending on your setup. See `backend/app.py` (or `auth.py`) to modify credentials.

---

## How to Use the App
1. **Log in** with Admin or Viewer account.
2. **Admin** can upload a CSV (see format below).
3. Use the **Filters** to narrow date range and category.
4. Explore charts:
   - **Pie**: item distribution by category.
   - **Bar**: total items per month.
   - **Line**: total revenue per month.
   - **Treemap (D3)**: revenue by category (size = revenue).
5. **Anomalies** will appear in the warning panel.
6. Click **Export Summary** to download the monthly summary CSV.

---

## CSV Upload Format

Upload a `.csv` with these columns (headers can be mapped, but defaults are below):

| Column       | Default Header | Type   | Example        |
|--------------|----------------|--------|----------------|
| Date         | `date`         | Date   | `2025-03-20`   |
| Category     | `category`     | String | `North`        |
| Item Count   | `value`        | Number | `104`          |
| Revenue      | `revenue`      | Number | `833.33`       |

**Sample CSV**
```csv
date,category,value,revenue
2025-03-01,North,104,833.33
2025-03-05,East,74,399.33
2025-03-09,South,20,123.38
```

> The frontend sends a `column_mapping` JSON when uploading:
```json
{
  "date_col": "date",
  "item_count_col": "value",
  "revenue_col": "revenue",
  "category_col": "category"
}
```

---

## API Endpoints
Base: `http://localhost:5001`

- `POST /api/login` → `{ username, password }` → `{ token, role }`
- `GET /api/data` → returns `{ raw_data, summary_data, anomaly_data }`
- `POST /api/upload` → multipart form: `file` (CSV), `column_mapping` (JSON)
- `GET /api/export` → downloads `summary_monthly.csv`

---

## Configuration

- **Frontend API Base**
  - In `frontend/src/App.js`, change:
    ```js
    const API_BASE = "http://localhost:5001";
    ```
  - For deployment, you can externalize to an env var (example): `REACT_APP_API_BASE=https://api.yourdomain.com`

- **Backend Port**
  - Default is `5001`. Change in `backend/app.py` if needed.

- **CORS**
  - Allow `http://localhost:3000` during development.
  - For production, restrict to your domain.

---

## Exporting Data
Click **Export Summary** in the UI (admin only) to download aggregated monthly metrics as CSV.

---

## Troubleshooting
- **Login fails** → Check backend is running and `API_BASE` is correct.
- **CORS error** → Enable CORS on backend and whitelist the frontend origin.
- **Upload failing** → Ensure CSV headers match or adjust `column_mapping`.
- **Blank charts** → No data or filters too restrictive; clear date filters.
- **Port conflicts** → Change frontend to 3001 or backend to 5002.

---

## Security Checklist
- Change default **admin/viewer** credentials.
- Do **not** commit `.env`, secrets, or CSVs with sensitive data.
- Limit CORS origins (production).
- Use HTTPS in production and rotate tokens regularly.

---

## Publish to GitHub (Step-by-step)

### A) Create the repo
1. Go to **github.com → New repository**.
2. Name it **`rama-omd-analytics`**.
3. Choose **Public** (simplest for full access) or **Private** then add collaborators.
4. Do **not** initialize with README (we have one). Click **Create repository**.

### B) Prepare your local folder
```bash
cd /path/to/your/project/root   # contains frontend/ and backend/
git init
git checkout -b main
# add a good .gitignore (see .gitignore file in this repo)
git add .
git commit -m "Initial commit: RAMA OMD Analytics"
```

### C) Connect remote & push
```bash
git remote add origin https://github.com/<YOUR-USERNAME>/rama-omd-analytics.git
git push -u origin main
```

> If prompted for a password and you have 2FA, use a **Personal Access Token**:
- GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens**
- Grant `contents: read/write` for your new repo.
- Use that token as the “password” in the terminal.

### D) Give the company full access
- **Public repo:** anyone can view/clone. (Use if they just need pull access.)
- **Private repo:** go to **Settings → Collaborators & teams → Add people**, enter their GitHub usernames, and set **Admin** or **Write** access as required.
- If they use a GitHub **Organization**, you can **transfer** the repo to that org from **Settings → Danger Zone → Transfer**.

---

## Hand-off Checklist
- [ ] Repo pushed to GitHub with README.md and .gitignore
- [ ] Company invited as collaborators **or** repo set to **Public**
- [ ] Backend and Frontend run locally with the instructions above
- [ ] Admin credentials changed and shared securely
- [ ] CSV sample provided and upload tested
- [ ] Export CSV verified
- [ ] (Optional) Domain & HTTPS configured for production

---

**Contact & Support**
- Issues can be opened in the GitHub repository under **Issues**.
- For deployment help, add a ticket or contact the maintainer.

