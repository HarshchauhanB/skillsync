# SkillSync AI — Smart Skill Gap Detector

A hackathon-ready Streamlit prototype that compares self-reported skills with sample role requirements and creates a practical learning roadmap.

## Features
- Student profile with target role and self-rated skill levels
- Role-specific skill-gap analysis
- TF-IDF and cosine-similarity demonstration
- 30-day roadmap with project suggestions
- SQLite-based progress tracker
- Downloadable roadmap CSV

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Create an app, select your repository and `main` branch, and set main file path to `app.py`.
4. Deploy.

## Important limitations
This is a prototype. Role skill maps are illustrative, skill levels are self-reported, and similarity is not a hiring probability. SQLite files on ephemeral hosting may not persist across restarts/redeployments. Do not enter sensitive personal data.
