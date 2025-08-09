# 📦 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.1.0] - 2025-07-11

### 🚀 Added
- Initial public release of **DeepClone**
- `POST /fetch` endpoint to clone and return the raw content of any public GitHub repository.
- Recursive file retrieval from nested folders using GitHub API.
- Automatic file-type distinction (file/dir) and recursive processing.
- Error handling for invalid URLs, missing files, and bad requests.
- Flask-based server running on port `5555`.
- `setup.py` script for automated setup:
  - Creates virtual environment.
  - Installs dependencies.
  - Runs the Flask app automatically.

### 🛠️ Tech Stack
- Python 3.12
- Flask
- Requests

---

## 🗂️ Structure
- `/fetch` — main API route
- `app.py` — core logic
- `setup.py` — environment bootstrapper
- `requirements.txt` — dependencies
- `setup-config.json` — customizable configuration
