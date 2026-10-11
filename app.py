import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "skillsync.db"

st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROLE_SKILLS = {
    "AI/ML Engineer": {
        "Python": 5, "Machine Learning": 5, "Statistics": 4,
        "SQL": 3, "Data Visualization": 3, "Git/GitHub": 3,
        "Deep Learning": 4, "Model Deployment": 3, "Data Cleaning": 4,
    },
    "Data Analyst": {
        "SQL": 5, "Python": 4, "Excel/Spreadsheets": 4,
        "Statistics": 4, "Data Visualization": 5, "Data Cleaning": 4,
        "Business Communication": 3, "Git/GitHub": 2,
    },
    "Python Developer": {
        "Python": 5, "Problem Solving": 4, "OOP": 4,
        "Git/GitHub": 4, "APIs": 4, "SQL": 3,
        "Testing/Debugging": 4, "Data Structures": 4,
    },
    "Frontend Developer": {
        "HTML": 5, "CSS": 5, "JavaScript": 5,
        "React": 4, "Git/GitHub": 3, "APIs": 3,
        "UI/UX Basics": 3, "Testing/Debugging": 2,
    },
    "Full-Stack Developer": {
        "HTML": 4, "CSS": 4, "JavaScript": 5,
        "React": 4, "Python": 3, "APIs": 4,
        "SQL": 4, "Git/GitHub": 4, "Testing/Debugging": 3,
    },
}

SKILL_DESCRIPTIONS = {
    "Python": "variables, functions, collections, files, and small scripts",
    "Machine Learning": "train/test split, classification, regression, and model evaluation",
    "Statistics": "mean, variance, probability, distributions, and hypothesis testing",
    "SQL": "SELECT, filtering, joins, grouping, and basic database design",
    "Data Visualization": "charts, dashboards, and communicating insights from data",
    "Git/GitHub": "commits, branches, pull requests, and project documentation",
    "Deep Learning": "neural networks, loss functions, training, and validation",
    "Model Deployment": "serving a model through a simple web app or API",
    "Data Cleaning": "missing values, duplicates, inconsistent formats, and validation",
    "Excel/Spreadsheets": "formulas, sorting, pivot tables, and data cleaning",
    "Business Communication": "explaining findings clearly and writing concise summaries",
    "Problem Solving": "breaking problems into steps and evaluating edge cases",
    "OOP": "classes, objects, inheritance, encapsulation, and composition",
    "APIs": "HTTP requests, JSON, REST endpoints, and error handling",
    "Testing/Debugging": "reproduce bugs, write tests, inspect errors, and verify fixes",
    "Data Structures": "lists, stacks, queues, hash maps, trees, and complexity basics",
    "HTML": "semantic page structure, forms, and accessible markup",
    "CSS": "layout, responsive design, flexbox/grid, and styling",
    "JavaScript": "variables, functions, DOM events, async code, and modules",
    "React": "components, props, state, hooks, and reusable UI",
    "UI/UX Basics": "user flows, wireframes, accessibility, and usability testing",
}

PROJECT_IDEAS = {
    "Python": "Build a command-line expense tracker with input validation.",
    "Machine Learning": "Train and evaluate a small classifier on a public dataset.",
    "Statistics": "Analyze a dataset and explain three statistically supported findings.",
    "SQL": "Create a small database and answer ten questions using SQL queries.",
    "Data Visualization": "Build a dashboard with three useful charts and written insights.",
    "Git/GitHub": "Publish a documented project with meaningful commits and a README.",
    "Deep Learning": "Train a small neural network and compare it with a baseline.",
    "Model Deployment": "Deploy a simple model behind a Streamlit interface.",
    "Data Cleaning": "Clean a messy CSV and document every transformation.",
    "Excel/Spreadsheets": "Create a pivot-table report from a sample sales dataset.",
    "Business Communication": "Write a one-page brief explaining a dataset's key findings.",
    "Problem Solving": "Solve and document 15 beginner programming problems.",
    "OOP": "Build a small library or inventory manager using classes.",
    "APIs": "Create an app that calls a public API and handles failures gracefully.",
    "Testing/Debugging": "Add unit tests and a bug-report checklist to an existing project.",
    "Data Structures": "Implement common data structures and compare operations.",
    "HTML": "Build a semantic portfolio page with a contact form.",
    "CSS": "Make a responsive landing page that works on mobile and desktop.",
    "JavaScript": "Build an interactive quiz with score tracking.",
    "React": "Create a reusable task manager with components and state.",
    "UI/UX Basics": "Prototype and test a three-screen student utility app.",
}

LEVELS = {"Beginner": 1, "Basic": 2, "Intermediate": 3, "Advanced": 4, "Proficient": 5}
LEVEL_OPTIONS = list(LEVELS.keys())

def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_key TEXT NOT NULL,
            skill TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            UNIQUE(student_key, skill)
        )
    """)
    conn.commit()
    return conn

def get_completed(student_key):
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT skill FROM progress WHERE student_key=? AND completed=1",
            (student_key,),
        ).fetchall()
    return {r[0] for r in rows}

def set_completed(student_key, skill, completed):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO progress(student_key, skill, completed, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_key, skill) DO UPDATE SET
                completed=excluded.completed, updated_at=excluded.updated_at
        """, (student_key, skill, int(completed), datetime.now().isoformat(timespec="seconds")))
        conn.commit()

def similarity_score(role, chosen_skills):
    required = list(ROLE_SKILLS[role].keys())
    if not chosen_skills:
        return {skill: 0.0 for skill in required}
    # TF-IDF is used to compare the student's self-described skill terms
    # with the role's required skill descriptions. Exact skill-level gaps
    # are shown separately and remain easy to interpret.
    student_text = " ".join(chosen_skills)
    role_text = " ".join(required)
    docs = [student_text, role_text]
    matrix = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b[\w+#./-]+\b").fit_transform(docs)
    overall = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    # Return an overall similarity reference value for each skill, not a
    # claimed probability or job-readiness score.
    return {skill: overall for skill in required}

def build_roadmap(role, current_levels):
    reqs = ROLE_SKILLS[role]
    gaps = []
    for skill, required_level in reqs.items():
        current_level = current_levels.get(skill, 0)
        if current_level < required_level:
            gaps.append((skill, required_level - current_level, required_level, current_level))
    gaps.sort(key=lambda item: (-item[1], item[0]))
    if not gaps:
        return [{
            "Day": "1–7", "Focus skill": "Portfolio & interview practice",
            "Learning task": "Review your strongest skills and explain your decisions.",
            "Mini-project / proof": "Publish a polished project README and demo.",
        }]
    roadmap = []
    for idx, (skill, gap, required, current) in enumerate(gaps[:10]):
        start = idx * 3 + 1
        end = min(start + 2, 30)
        roadmap.append({
            "Day": f"{start}–{end}",
            "Focus skill": skill,
            "Learning task": f"Practice {SKILL_DESCRIPTIONS.get(skill, skill)}.",
            "Mini-project / proof": PROJECT_IDEAS.get(skill, f"Build a small project demonstrating {skill}."),
        })
    if len(roadmap) < 10:
        roadmap.append({
            "Day": "28–30",
            "Focus skill": "Review & showcase",
            "Learning task": "Review your work, write what you learned, and identify next steps.",
            "Mini-project / proof": "Publish your best project with screenshots, README, and a short demo.",
        })
    return roadmap

if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "branch": "AIML",
        "role": "AI/ML Engineer",
        "skills": [],
        "levels": {},
        "hours": 1,
    }

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #14213d 0%, #284b63 55%, #3c6e71 100%);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {font-size: 2.35rem; margin: 0; color: white;}
    .hero p {font-size: 1.05rem; margin: .5rem 0 0 0; color: #edf6f9;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎯 SkillSync AI")
    st.caption("Your personal skill-gap and learning companion")
    page = st.radio(
        "Navigate",
        ["Overview", "Build my profile", "Skill gap analysis", "30-day roadmap", "Progress tracker", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("Prototype • Self-reported skills • Not a hiring guarantee")

profile = st.session_state.profile
student_key = (profile.get("name", "").strip().lower() or "demo-student")[:80]

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <h1>SkillSync AI 🎯</h1>
      <p>Understand your skill gaps. Build a focused plan. Prove what you can do.</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Your career-growth workspace")
    st.write("Choose a target role and describe your current skills. SkillSync compares your self-assessment with a transparent role-skill map, then suggests practical next steps.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Career roles", len(ROLE_SKILLS))
    c2.metric("Tracked skills", len(SKILL_DESCRIPTIONS))
    c3.metric("Roadmap length", "30 days")
    st.markdown("### Start here")
    st.write("1. **Build my profile** — add your career goal and skill levels.")
    st.write("2. **Skill gap analysis** — review your missing and developing skills.")
    st.write("3. **30-day roadmap** — get small, project-based practice tasks.")
    st.write("4. **Progress tracker** — mark learning tasks complete.")
    st.info("Demo tip: use the sample profile in the profile page, then show the skill gap and roadmap to the judges.")
    if st.button("Load a sample AIML student profile", type="primary"):
        st.session_state.profile = {
            "name": "Demo Student",
            "branch": "AIML",
            "role": "AI/ML Engineer",
            "skills": ["Python", "Data Visualization", "Git/GitHub"],
            "levels": {"Python": "Intermediate", "Data Visualization": "Basic", "Git/GitHub": "Basic"},
            "hours": 2,
        }
        st.rerun()

elif page == "Build my profile":
    st.markdown("## 👤 Build your student profile")
    st.write("Use honest self-ratings. You can update them as you learn.")
    all_skills = sorted(SKILL_DESCRIPTIONS.keys())
    with st.form("profile_form"):
        name = st.text_input("Name or demo nickname", value=profile.get("name", ""), max_chars=80)
        branch = st.selectbox("Branch / stream", ["AIML", "CSE", "ISE", "ECE", "Other"], index=["AIML", "CSE", "ISE", "ECE", "Other"].index(profile.get("branch", "AIML")))
        role_names = list(ROLE_SKILLS.keys())
        role = st.selectbox("Target job role", role_names, index=role_names.index(profile.get("role", "AI/ML Engineer")))
        hours = st.slider("Time available for learning each day", 1, 6, int(profile.get("hours", 1)))
        skills = st.multiselect("Skills you have tried", all_skills, default=[s for s in profile.get("skills", []) if s in all_skills])
        st.markdown("### Rate your current level")
        levels = {}
        if skills:
            cols = st.columns(2)
            old_levels = profile.get("levels", {})
            for i, skill in enumerate(skills):
                with cols[i % 2]:
                    old = old_levels.get(skill, "Beginner")
                    levels[skill] = st.select_slider(
                        skill, options=LEVEL_OPTIONS,
                        value=old if old in LEVEL_OPTIONS else "Beginner",
                        key=f"level_{skill}",
                    )
        submitted = st.form_submit_button("Save profile", type="primary", use_container_width=True)
        if submitted:
            st.session_state.profile = {
                "name": name.strip(),
                "branch": branch,
                "role": role,
                "skills": skills,
                "levels": levels,
                "hours": hours,
            }
            st.success("Profile saved!")
            st.rerun()
    st.caption("Skill levels: Beginner = 1, Basic = 2, Intermediate = 3, Advanced = 4, Proficient = 5.")

elif page == "Skill gap analysis":
    st.markdown("## 🧠 Skill gap analysis")
    role = profile.get("role", "AI/ML Engineer")
    current_levels = {s: LEVELS.get(l, 1) for s, l in profile.get("levels", {}).items()}
    reqs = ROLE_SKILLS[role]
    if not profile.get("skills"):
        st.warning("Your profile has no skills yet. Add a few in **Build my profile**, or load the sample profile from Overview.")
    else:
        rows = []
        for skill, required in reqs.items():
            current = current_levels.get(skill, 0)
            gap = max(required - current, 0)
            rows.append({
                "Skill": skill,
                "Your level": current,
                "Suggested target": required,
                "Gap": gap,
                "Status": "Meets target" if gap == 0 else ("Developing" if current else "Not added"),
            })
        df = pd.DataFrame(rows)
        covered = sum(min(r["Your level"], r["Suggested target"]) for _, r in df.iterrows())
        total = sum(reqs.values())
        coverage = int(round(100 * covered / total)) if total else 0
        gaps = df[df["Gap"] > 0].sort_values(["Gap", "Skill"], ascending=[False, True])
        a, b, c = st.columns(3)
        a.metric("Skill coverage", f"{coverage}%")
        b.metric("Skills to develop", int((df["Gap"] > 0).sum()))
        c.metric("Role target skills", len(reqs))
        st.progress(coverage / 100, text="Coverage against the role's suggested skill levels")
        fig = px.bar(
            df.sort_values("Gap", ascending=True), x="Gap", y="Skill", orientation="h",
            title=f"Skill gaps for {role}", color="Gap",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=420, coloraxis_showscale=False, xaxis_title="Level difference (0–5)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("### Recommended priorities")
        if gaps.empty:
            st.success("All listed skills meet or exceed the suggested levels. Focus on portfolio quality and interview practice.")
        else:
            for _, row in gaps.head(5).iterrows():
                skill = row["Skill"]
                st.markdown(f"**{skill}** — {row['Status']} (gap: {row['Gap']})")
                st.caption(SKILL_DESCRIPTIONS.get(skill, "Practice the fundamentals and apply them in a small project."))
        st.caption("This is a planning aid based on your self-rating and a sample role-skill map, not a validated hiring prediction.")
        with st.expander("How the text-matching component works"):
            text = " ".join(profile.get("skills", []))
            required_text = " ".join(reqs.keys())
            try:
                mat = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([text, required_text])
                score = float(cosine_similarity(mat[0:1], mat[1:2])[0, 0])
                st.write(f"TF-IDF / cosine similarity reference: **{score:.2f}**")
                st.write("This indicates textual overlap between the entered skill names and the role's required skill names. It is not a probability of getting hired; the skill-level gap chart is the main recommendation signal.")
            except ValueError:
                st.write("Add skill names to calculate textual overlap.")

elif page == "30-day roadmap":
    st.markdown("## 🗓️ Your 30-day roadmap")
    role = profile.get("role", "AI/ML Engineer")
    current_levels = {s: LEVELS.get(l, 1) for s, l in profile.get("levels", {}).items()}
    roadmap = build_roadmap(role, current_levels)
    st.write(f"**Target role:** {role} · **Daily time:** {profile.get('hours', 1)} hour(s)")
    st.info("Treat the roadmap as a flexible guide. Adjust the pace to your timetable and prior knowledge.")
    roadmap_df = pd.DataFrame(roadmap)
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
    st.markdown("### Suggested weekly rhythm")
    st.write("- **Week 1:** Learn fundamentals and take short notes.")
    st.write("- **Week 2:** Practice with small exercises.")
    st.write("- **Week 3:** Build a mini-project using the skill.")
    st.write("- **Week 4:** Improve the project, document it, and share the result.")
    st.download_button(
        "Download roadmap as CSV",
        data=roadmap_df.to_csv(index=False).encode("utf-8"),
        file_name="skillsync_30_day_roadmap.csv",
        mime="text/csv",
        use_container_width=True,
    )

elif page == "Progress tracker":
    st.markdown("## ✅ Progress tracker")
    role = profile.get("role", "AI/ML Engineer")
    current_levels = {s: LEVELS.get(l, 1) for s, l in profile.get("levels", {}).items()}
    roadmap = build_roadmap(role, current_levels)
    skills = list(dict.fromkeys([r["Focus skill"] for r in roadmap]))
    completed = get_completed(student_key)
    if not skills:
        st.write("Build your profile first to create your learning tasks.")
    else:
        st.write("Check off tasks after you have actually completed the learning or project evidence.")
        done_count = sum(s in completed for s in skills)
        st.metric("Roadmap tasks completed", f"{done_count}/{len(skills)}")
        st.progress(done_count / len(skills))
        for skill in skills:
            is_done = skill in completed
            new_val = st.checkbox(f"{skill}", value=is_done, key=f"progress_{student_key}_{skill}")
            if new_val != is_done:
                set_completed(student_key, skill, new_val)
                st.rerun()
        st.caption("Progress is saved in a local SQLite database. On free cloud hosting, local database files may not persist through every restart or redeployment.")
        if st.button("Reset this profile's progress"):
            for skill in skills:
                set_completed(student_key, skill, False)
            st.rerun()

else:
    st.markdown("## ℹ️ About SkillSync AI")
    st.write("SkillSync AI is a student career-planning prototype that compares self-reported skills with an editable sample map of role requirements.")
    st.markdown("### ML / logic used")
    st.write("- **TF-IDF:** converts skill descriptions into weighted text features.")
    st.write("- **Cosine similarity:** estimates textual overlap between student skills and target-role skills.")
    st.write("- **Skill-level gap analysis:** compares self-rated levels with suggested levels for each target role.")
    st.write("- **Rule-based roadmap:** orders developing skills and links them to practical learning tasks.")
    st.markdown("### Limitations")
    st.write("- Role requirements are example data, not scraped live job listings.")
    st.write("- Skill ratings are self-reported and may be inaccurate.")
    st.write("- Similarity scores are not hiring probabilities or guarantees.")
    st.write("- Do not enter passwords, government IDs, private documents, or other sensitive personal data.")
    st.markdown("### Suggested future improvements")
    st.write("Connect real job descriptions, add a verified course catalog, export PDF reports, support faculty/mentor feedback, and use persistent cloud storage.")
