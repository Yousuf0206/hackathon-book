# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document defines the key entities and their relationships for the Physical AI & Humanoid Robotics book project. Since this is a documentation project rather than a traditional software application, the "data model" represents the structural elements of the book content and associated metadata.

## Key Entities

### Book
- **Name**: Physical AI & Humanoid Robotics Book
- **Description**: Comprehensive educational resource covering ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA technologies for humanoid robotics
- **Fields**:
  - title: string
  - version: string
  - publication_date: date
  - total_modules: integer
  - total_chapters: integer
  - word_count: integer
  - target_readability: string (Flesch-Kincaid Grade Level)
  - citation_style: string (APA 7th edition)
  - deployment_url: string
  - pdf_path: string

### Module
- **Name**: Book Module
- **Description**: Major section of the book covering a specific technology area
- **Fields**:
  - id: string (e.g., "module1-ros2")
  - title: string
  - description: string
  - order: integer
  - chapters_count: integer
  - estimated_duration: integer (hours)
  - learning_objectives: array of strings
  - prerequisites: array of strings
  - technologies_covered: array of strings

### Chapter
- **Name**: Book Chapter
- **Description**: Individual chapter within a module containing specific learning content
- **Fields**:
  - id: string (e.g., "module1-chapter1")
  - title: string
  - description: string
  - order: integer
  - module_id: string
  - word_count: integer
  - reading_time: integer (minutes)
  - learning_objectives: array of strings
  - key_concepts: array of strings
  - code_examples_count: integer
  - diagrams_count: integer
  - exercises_count: integer

### CodeExample
- **Name**: Code Example
- **Description**: Reproducible code snippet included in chapters
- **Fields**:
  - id: string
  - title: string
  - description: string
  - language: string (python, cpp, xml, etc.)
  - code_content: string
  - chapter_id: string
  - file_path: string
  - dependencies: array of strings
  - expected_output: string
  - validation_status: enum (unverified, verified, needs_fix)

### Diagram
- **Name**: Diagram/Visual Element
- **Description**: Visual representation of concepts, architecture, or processes
- **Fields**:
  - id: string
  - title: string
  - description: string
  - type: string (flowchart, architecture, process, etc.)
  - file_path: string
  - chapter_id: string
  - alt_text: string
  - source_tool: string (mermaid, draw.io, blender, etc.)

### Citation
- **Name**: Academic Citation
- **Description**: Reference to external source following APA format
- **Fields**:
  - id: string
  - apa_format: string
  - bibtex_key: string
  - title: string
  - authors: array of strings
  - publication_year: integer
  - source_type: enum (journal, conference, book, website, documentation)
  - url: string (optional)
  - accessed_date: date (optional)
  - peer_reviewed: boolean
  - used_in_chapters: array of strings
  - verification_status: enum (unverified, verified, needs_review)

### Exercise
- **Name**: Learning Exercise
- **Description**: Practical task for students to reinforce learning
- **Fields**:
  - id: string
  - title: string
  - description: string
  - difficulty_level: enum (beginner, intermediate, advanced)
  - estimated_duration: integer (minutes)
  - chapter_id: string
  - solution_available: boolean
  - solution_path: string (optional)
  - learning_objectives_addressed: array of strings

## Relationships

### Book → Module
- One-to-Many relationship
- A book contains multiple modules
- Each module belongs to one book

### Module → Chapter
- One-to-Many relationship
- A module contains multiple chapters
- Each chapter belongs to one module

### Chapter → CodeExample
- One-to-Many relationship
- A chapter may contain multiple code examples
- Each code example belongs to one chapter

### Chapter → Diagram
- One-to-Many relationship
- A chapter may contain multiple diagrams
- Each diagram belongs to one chapter

### Chapter → Exercise
- One-to-Many relationship
- A chapter may contain multiple exercises
- Each exercise belongs to one chapter

### Citation → Chapter
- Many-to-Many relationship
- A citation may be used across multiple chapters
- A chapter may reference multiple citations

## Validation Rules

### Book Validation
- Total modules must match defined modules (4 for this book)
- Target readability must be between Grade 10-12 (Flesch-Kincaid)
- Citation style must be APA 7th edition
- Word count must be between 5,000-7,000 words per module

### Module Validation
- Module order must be sequential (1, 2, 3, 4)
- Module ID must follow pattern: "module[1-9]+-[a-z-]+"
- Each module must have 4-6 chapters
- Learning objectives must be specific and measurable

### Chapter Validation
- Chapter order within module must be sequential
- Chapter ID must follow pattern: "module[1-9]+-chapter[1-9]+"
- Word count should be between 800-1500 words
- Must include at least 3 code examples (for Module 1) or 1 diagram (for Module 2)
- Learning objectives must align with module objectives

### Code Example Validation
- Language must be specified
- Code content must be syntactically valid
- Dependencies must be documented
- Validation status must be verified before publication

### Citation Validation
- Must have APA format string
- Must specify source type
- At least 50% must be peer-reviewed sources
- Must be linked to at least one chapter
- Verification status must be verified

## State Transitions

### Chapter State Model
- draft → review → revision → verified → published
- draft: Initial content created
- review: Content under review by technical expert
- revision: Content being updated based on feedback
- verified: Content verified for accuracy and clarity
- published: Content published in final book

### Code Example State Model
- unverified → tested → verified → integrated
- unverified: Code example created but not tested
- tested: Code example tested in appropriate environment
- verified: Code example confirmed working as expected
- integrated: Code example integrated into chapter

## Content Workflows

### Writing Workflow
1. Research phase: Gather sources and plan content
2. Draft phase: Create initial content following chapter template
3. Technical review: Verify code examples and technical accuracy
4. Content review: Check clarity, readability, and flow
5. Citation verification: Ensure all sources are properly cited
6. Final verification: Complete quality checks before publication

### Quality Assurance Workflow
1. Readability check: Verify Flesch-Kincaid grade level
2. Plagiarism check: Ensure original content with proper attribution
3. Technical validation: Test all code examples and procedures
4. Citation validation: Verify all sources and APA formatting
5. Accessibility check: Ensure alt text and semantic structure