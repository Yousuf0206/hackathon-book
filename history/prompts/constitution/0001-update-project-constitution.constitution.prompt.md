---
id: 0001
title: Update Project Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: 001-physical-ai-book
user: unknown
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:

---

## Prompt

```text
/sp.constitution
## Project: AI/Spec-Driven Book Creation Using Docusaurus, Spec-Kit Plus, and Claude Code

---

## Core Principles

### 1. **Accuracy**
- All factual statements must be verified using primary or authoritative sources.
- No speculative claims or unverifiable information.
- Every referenced fact must be traceable.

### 2. **Clarity**
- Target audience: **Academic readers with a computer science background**.
- Maintain clear, structured explanations.
- Required readability: **Flesch-Kincaid Grade 10–12**.

### 3. **Reproducibility**
- All processes, technical explanations, and claims must be reproducible.
- Readers should be able to follow instructions to replicate findings.
- Sources must be explicitly cited and discoverable.

### 4. **Rigor**
- Prefer **peer-reviewed** literature when possible.
- Maintain academic tone and evidence-driven writing.
- Use established research and industry standards.

---

## Key Standards

### **Citation Requirements**
- Format: **APA (7th edition)**.
- Minimum **50% peer-reviewed sources**.
- Acceptable supplemental sources:
  - Whitepapers
  - Technical specifications
  - Framework documentation (ROS 2, Isaac, Unity, etc.)
  - Scientific blogs or official repositories

### **Plagiarism**
- **Zero-tolerance** policy.
- Must pass plagiarism checking tools before submission.

### **Writing Style**
- Academic, neutral, and precise.
- No marketing language or exaggerated claims.
- Consistent vocabulary, formatting, and structure across all chapters.

---

## Constraints

- **Word Count:** 5,000–7,000 words
- **Minimum Sources:** 15
- **Output Format:**
  - Docusaurus site deployed to GitHub Pages
  - PDF export with **embedded APA citations**

- **Tools Used:**
  - Spec-Kit Plus
  - Claude Code
  - GitHub
  - Docusaurus

---

## Success Criteria

### 1. **Verification**
- All factual statements are backed by properly formatted APA citations.
- Every claim passes a source traceability check.

### 2. **Originality**
- No plagiarism detected by au- `/sp.chapters`
- Individual chapter specs (`/sp.chapter.*`)

All future specs must strictly follow the principles defined here.

---
```

## Response snapshot

Updated the project constitution at `.specify/memory/constitution.md` with the provided content, including core principles, key standards, constraints, and success criteria. The constitution is version 1.0.0, ratified and last amended on 2025-12-07. A Sync Impact Report was prepended to the file.

## Outcome

- ✅ Impact: Project constitution updated.
- 🧪 Tests: None required.
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next prompts: None
- 🧠 Reflection: Successfully updated the constitution file and generated a PHR.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): None
- Prompt variant (if applicable): None
- Next experiment (smallest change to try): None
