# AI/Spec-Driven Book Creation Using Docusaurus, Spec-Kit Plus, and Claude Code Constitution
<!-- Sync Impact Report:
Version change: None -> 1.0.0
Modified principles: None
Added sections: Accuracy, Clarity, Reproducibility, Rigor, Key Standards, Constraints, Success Criteria
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ✅ updated
- .specify/templates/spec-template.md: ✅ updated
- .specify/templates/tasks-template.md: ✅ updated
- .specify/templates/commands/*.md: ✅ updated
- runtime guidance docs (README.md, docs/quickstart.md): ✅ updated
Follow-up TODOs: None
-->

## Core Principles

### 1. Accuracy
- All factual statements must be verified using primary or authoritative sources.
- No speculative claims or unverifiable information.
- Every referenced fact must be traceable.

### 2. Clarity
- Target audience: **Academic readers with a computer science background**.
- Maintain clear, structured explanations.
- Required readability: **Flesch-Kincaid Grade 10–12**.

### 3. Reproducibility
- All processes, technical explanations, and claims must be reproducible.
- Readers should be able to follow instructions to replicate findings.
- Sources must be explicitly cited and discoverable.

### 4. Rigor
- Prefer **peer-reviewed** literature when possible.
- Maintain academic tone and evidence-driven writing.
- Use established research and industry standards.

## Key Standards

### Citation Requirements
- Format: **APA (7th edition)**.
- Minimum **50% peer-reviewed sources**.
- Acceptable supplemental sources:
  - Whitepapers
  - Technical specifications
  - Framework documentation (ROS 2, Isaac, Unity, etc.)
  - Scientific blogs or official repositories

### Plagiarism
- **Zero-tolerance** policy.
- Must pass plagiarism checking tools before submission.

### Writing Style
- Academic, neutral, and precise.
- No marketing language or exaggerated claims.
- Consistent vocabulary, formatting, and structure across all chapters.

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

## Success Criteria

### 1. Verification
- All factual statements are backed by properly formatted APA citations.
- Every claim passes a source traceability check.

### 2. Originality
- No plagiarism detected by au- `/sp.chapters`
- Individual chapter specs (`/sp.chapter.*`)

## Governance

All future specs must strictly follow the principles defined here.

**Version**: 1.0.0 | **Ratified**: 2025-12-07 | **Last Amended**: 2025-12-07
