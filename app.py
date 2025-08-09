from flask import Flask, request, jsonify, Response
import requests
import os
import json
import hashlib
from pathlib import Path
from typing import Tuple, Optional
import json


app = Flask(__name__)

# GitHub API base
GITHUB_API = "https://api.github.com/repos"

# Cache config
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------- Utils / Cache ----------------------------

def _cache_key(repo_url: str, branch: Optional[str]) -> str:
    """
    make cache key sensitive to both repo_url (may include subpath) and branch.
    """
    base = f"{repo_url}|{branch or ''}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def get_cache_path(repo_url: str, branch: Optional[str]) -> str:
    return os.path.join(CACHE_DIR, f"{_cache_key(repo_url, branch)}.json")


def load_from_cache(repo_url: str, branch: Optional[str]):
    path = get_cache_path(repo_url, branch)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(repo_url: str, branch: Optional[str], data):
    path = get_cache_path(repo_url, branch)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_repo_input(value: str) -> Tuple[str, str, str, str]:
    """
    Accepts:
      - https://github.com/USER/REPO[/sub/path]
      - USER/REPO[/sub/path]
    Returns (user, repo, subpath, normalized_url_for_cache)
    """
    v = value.strip()

    if "github.com" in v:
        v = v.replace("https://github.com/", "").replace("http://github.com/", "")
    v = v.strip("/")

    # strip trailing .git
    if v.endswith(".git"):
        v = v[:-4]

    parts = v.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid repo format. Use https://github.com/USER/REPO or USER/REPO")

    user, repo = parts[0], parts[1]
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

    # normalized url (used for cache key)
    norm = f"https://github.com/{user}/{repo}"
    if subpath:
        norm = f"{norm}/{subpath}"

    return user, repo, subpath, norm


# ---------------------------- GitHub fetch ----------------------------

def fetch_files_from_github(user: str, repo: str, path: str = "", branch: Optional[str] = None, token: Optional[str] = None):
    """
    Recursively fetch files content under 'path' on given 'branch' (default repo branch if None).
    """
    params = {}
    if branch:
        params["ref"] = branch

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        # token can be "ghp_xxx" or "Bearer xxx" or "token xxx" (we just pass through)
        if not token.lower().startswith(("token ", "bearer ")):
            headers["Authorization"] = f"token {token}"
        else:
            headers["Authorization"] = token

    url = f"{GITHUB_API}/{user}/{repo}/contents/{path}".rstrip("/")
    r = requests.get(url, params=params, headers=headers)
    if r.status_code != 200:
        return None, r.status_code, r.text

    content = r.json()
    files_data = {}

    for item in content:
        if item["type"] == "file":
            file_resp = requests.get(item["download_url"], headers=headers)
            file_resp.raise_for_status()
            files_data[item["path"]] = file_resp.text
        elif item["type"] == "dir":
            sub_files, code, _ = fetch_files_from_github(
                user, repo, item["path"], branch=branch, token=token
            )
            if code == 200 and sub_files:
                files_data.update(sub_files)
            else:
                # bubble up error if directory listing failed
                return None, code, f"Failed to list {item['path']}"
        # ignore symlinks/submodules for simplicity

    return files_data, 200, "OK"


# ---------------------------- Core ----------------------------


def core_fetch(repo_like: str, branch=None, path_override=None, token=None):
    user, repo, subpath, norm_url = parse_repo_input(repo_like)
    if path_override:
        subpath = f"{subpath}/{path_override}".strip("/")

    # لا نستخدم الكاش إذا كان فيه توكن (احتمال ريبو خاص)
    if not token:
        cached = load_from_cache(norm_url, branch)
        if cached:
            return cached, 200

    # fallback لتوكن البيئة فقط إن لم يُرسل بالطلب
    token = token or os.environ.get("GITHUB_TOKEN")

    files_data, code, msg = fetch_files_from_github(user, repo, subpath, branch=branch, token=token)
    if code != 200:
        return {"error": msg}, code

    result = {"files": files_data, "repo": f"{user}/{repo}", "branch": branch or "default", "path": subpath}

    if not token:  # لا نكتب محتوى خاص على القرص
        save_to_cache(norm_url, branch, result)
    return result, 200



# ---------------------------- Routes ----------------------------

@app.post("/fetch")
def fetch_post():
    data = request.get_json(silent=True) or {}
    repo_like = data.get("repo") or data.get("repo_url")
    branch = data.get("branch")
    path_override = data.get("path")

    if not repo_like:
        return jsonify({"error": "repo is required"}), 400

    try:
        result, code = core_fetch(repo_like, branch=branch, path_override=path_override)

        # ← نفس اللوج هنا أيضًا لو بدك
        files = result.get("files", {}) if isinstance(result, dict) else {}
        app.logger.info(
            "Fetched %d files from %s (branch=%s, path=%s)",
            len(files),
            result.get("repo"),
            result.get("branch"),
            result.get("path"),
        )

        return jsonify(result), code
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.get("/fetch")
def fetch_get():
    repo_like = request.args.get("repo") or request.args.get("repo_url")
    branch = request.args.get("branch")
    path_override = request.args.get("path")

    if not repo_like:
        return jsonify({"error": "repo is required"}), 400

    try:
        result, code = core_fetch(repo_like, branch=branch, path_override=path_override)

        # ← أضف اللوج هنا (بعد core_fetch وقبل return)
        files = result.get("files", {}) if isinstance(result, dict) else {}
        app.logger.info(
            "Fetched %d files from %s (branch=%s, path=%s)",
            len(files),
            result.get("repo"),
            result.get("branch"),
            result.get("path"),
        )

        return jsonify(result), code
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    # Debug server (dev only)
    app.run(host="127.0.0.1", port=5555, debug=False)
    # For production, use a WSGI server like Gunicorn or uWSGI
    #app.run(host="127.0.0.1", port=5555, debug=True, use_reloader=True)
