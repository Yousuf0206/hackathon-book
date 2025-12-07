# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Overview
This guide provides a quick setup for contributing to the Physical AI & Humanoid Robotics book project. The book covers ROS 2, Gazebo, Unity, NVIDIA Isaac, and VLA technologies for humanoid robotics using Docusaurus.

## Prerequisites
- Node.js 18+ with npm or yarn
- Git
- Python 3.8+ (for ROS 2 examples)
- Basic understanding of robotics concepts

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Docusaurus Dependencies
```bash
npm install
# or
yarn install
```

### 3. Start Local Development Server
```bash
npm run start
# or
yarn start
```

This command starts a local development server and opens the documentation in your browser. Most changes are reflected live without restarting the server.

### 4. Build for Production
```bash
npm run build
# or
yarn build
```

This command generates static content into the `build` directory and can be served using any static hosting service.

## Project Structure
```
docs/
├── modules/              # Book modules (1-4)
│   ├── module1-ros2/     # ROS 2 fundamentals
│   ├── module2-digital-twin/  # Gazebo & Unity
│   ├── module3-ai-brain/      # NVIDIA Isaac
│   └── module4-vla/           # Vision-Language-Action
├── assets/               # Images, diagrams, code examples
├── bibliography/         # Reference materials
└── _category_.json       # Navigation configuration
```

## Writing Content

### Creating a New Chapter
1. Create a new directory in the appropriate module folder
2. Add a `README.md` file with the chapter content
3. Update the `_category_.json` file to include the new chapter in the navigation

### Adding Code Examples
- Place code examples in the `assets/code-examples/` directory
- Use Docusaurus' built-in code block syntax with appropriate language tags
- Test all examples in appropriate environments before adding

### Adding Citations
- Follow APA 7th edition format
- Add citations to the `bibliography/references.md` file
- Use inline citations in the format `[Author, Year]` or as appropriate for APA style

## Quality Standards

### Readability
- Maintain Flesch-Kincaid Grade Level 10-12
- Use clear, concise language
- Define technical terms when first used

### Technical Accuracy
- Verify all code examples work as described
- Test installation instructions in clean environments
- Ensure all technical claims are accurate

### Citation Requirements
- Minimum 50% peer-reviewed sources
- Follow APA 7th edition format
- Verify all sources are accessible and valid

## Building the PDF
The PDF version is automatically generated during the build process and placed in the `static/pdf/` directory. Ensure all content renders properly for print format.

## Deployment
The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment workflow includes:
- Build verification
- Content quality checks
- PDF generation
- Publication to GitHub Pages