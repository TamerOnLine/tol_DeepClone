"""Flask service for listing and fetching files from GitHub repositories.

This module exposes two HTTP endpoints:

- ``GET /tree``  — list repository files (optionally filtered by path/patterns).
- ``GET|POST /fetch`` — fetch file metadata and, optionally, contents.

The code includes simple, file-based caching to reduce GitHub API calls.
All comments and documentation are in English, and the code follows PEP 8.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import time
import zipfile
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, Response, jsonify, request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


app = Flask(__name__)


# ---------------------------- Config ----------------------------
GITHUB_API = "https://api.github.com/repos"
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Default TTL is 30 minutes
CACHE_TTL = int(os.getenv("CACHE_TTL", "1800"))
MAX_FILES_DEFAULT = int(os.getenv("MAX_FILES", "1000"))
MAX_BYTES_DEFAULT = int(os.getenv("MAX_TOTAL_BYTES", "5000000"))

# Optional (Authorization: Bearer <API_KEY>)
API_KEY = os.getenv("API_KEY")


# ---------------------------- Helpers ----------------------------
TEXT_EXTS = {".toml", ".md", ".yml", ".yaml", ".ini", ".cfg", ".env", ".txt"}
TEXT_FILENAMES = {
    "license", "license.txt", "notice", "readme", "readme.md",
    ".gitignore", ".gitattributes", ".editorconfig"
}


def require_api_key() -> Optional[Tuple[Response, int]]:
    """Validate the API key from the ``Authorization`` header.

    Returns:
        Optional[Tuple[Response, int]]: ``None`` if access is allowed; otherwise,
        a tuple of (JSON error response, HTTP status code 401).

    Notes:
        Expects ``Authorization: Bearer <API_KEY>`` or ``Authorization: <API_KEY>``.
    """
    if not API_KEY:
        return None

    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else auth
    if token != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    return None


def parse_repo_input(value: str) -> Tuple[str, str]:
    """Normalize a GitHub repository input value.

    Args:
        value (str): Input such as ``"USER/REPO"`` or a GitHub URL.

    Returns:
        Tuple[str, str]: A pair ``(normalized_repo, subpath)`` where
        ``normalized_repo`` is ``"USER/REPO"`` and ``subpath`` is any remaining
        path segment after the repo portion (possibly empty).

    Raises:
        ValueError: If the input does not contain at least ``USER/REPO``.
    """
    v = (value or "").strip()
    if "github.com" in v:
        v = v.replace("https://github.com/", "").replace("http://github.com/", "")
    v = v.strip("/")
    if v.endswith(".git"):
        v = v[:-4]

    parts = v.split("/")
    if len(parts) < 2:
        raise ValueError(
            "Invalid repo format. Use https://github.com/USER/REPO or USER/REPO"
        )

    user, repo = parts[0], parts[1]
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""
    norm = f"{user}/{repo}"
    return norm, subpath


def make_session(token: Optional[str]) -> requests.Session:
    """Create a configured ``requests.Session`` for GitHub API access."""
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-mvp/1.0",
    }
    if token:
        if not token.lower().startswith(("token ", "bearer ")):
            headers["Authorization"] = f"token {token}"
        else:
            headers["Authorization"] = token
    session.headers.update(headers)

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def csv_patterns(value: Optional[str]) -> List[str]:
    """Split a comma-separated string of glob patterns into a list."""
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def match_any(path: str, patterns: List[str]) -> bool:
    """Return True if the path matches any of the given glob patterns."""
    return any(fnmatch(path, pattern) for pattern in patterns)


def filter_files(
    files: List[Dict[str, Any]],
    include: List[str],
    exclude: List[str],
    prefix: str,
) -> List[Dict[str, Any]]:
    """Filter a Git tree listing by prefix and include/exclude patterns."""
    out: List[Dict[str, Any]] = []
    normalized_prefix = prefix.strip("/")

    for f in files:
        path = f["path"]

        if normalized_prefix and not (
            path == normalized_prefix or path.startswith(normalized_prefix + "/")
        ):
            continue
        if include and not match_any(path, include):
            continue
        if exclude and match_any(path, exclude):
            continue

        out.append(f)

    return out


def is_binary_mime(mime: str) -> bool:
    """Heuristically determine whether a MIME type is binary."""
    if not mime:
        return True
    if mime.startswith("text/") or "json" in mime or "xml" in mime or "javascript" in mime:
        return False
    return True


def guess_mime(path: str) -> str:
    """Guess the MIME type for a path."""
    name = os.path.basename(path).lower()
    if name in TEXT_FILENAMES:
        return "text/plain"
    ext = os.path.splitext(name)[1]
    if ext in TEXT_EXTS:
        return "text/plain"
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


# ---------------------------- Cache (file TTL) ----------------------------
def cache_key(kind: str, **kwargs: Any) -> str:
    """Construct a deterministic cache key for a payload type and parameters."""
    raw = kind + "|" + "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path(key: str) -> str:
    """Return the on-disk cache file path for a given key."""
    return os.path.join(CACHE_DIR, f"{key}.json")


def cache_get(key: str) -> Any:
    """Load a cached payload if it exists and has not expired."""
    path = cache_path(key)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if time.time() - obj.get("_ts", 0) <= CACHE_TTL:
            return obj.get("payload")
    except Exception:
        return None

    return None


def cache_set(key: str, payload: Any) -> None:
    """Persist a payload in the cache with a timestamp."""
    path = cache_path(key)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_ts": time.time(), "payload": payload}, fh, ensure_ascii=False)


# ---------------------------- GitHub API wrappers ----------------------------
def get_tree(session: requests.Session, repo: str, ref: str) -> List[Dict[str, Any]]:
    """Retrieve the Git tree for a repository and reference."""
    # Resolve HEAD early to avoid a 404 + retry
    if ref.upper() == "HEAD":
        meta = session.get(f"{GITHUB_API}/{repo}", timeout=(5, 15))
        meta.raise_for_status()
        ref = meta.json().get("default_branch", "main")

    url = f"{GITHUB_API}/{repo}/git/trees/{ref}"
    response = session.get(url, params={"recursive": "1"}, timeout=(5, 30))

    if response.status_code == 404:
        # The ref might not exist. Try the repository default branch from /repos.
        meta = session.get(f"{GITHUB_API}/{repo}", timeout=(5, 15))
        meta.raise_for_status()
        default_ref = meta.json().get("default_branch", "main")
        response = session.get(
            f"{GITHUB_API}/{repo}/git/trees/{default_ref}",
            params={"recursive": "1"},
            timeout=(5, 30),
        )

    response.raise_for_status()
    tree = response.json().get("tree", [])
    return [
        {"path": t["path"], "sha": t.get("sha"), "size": t.get("size", 0)}
        for t in tree
        if t.get("type") == "blob"
    ]


def fetch_blob(session: requests.Session, repo: str, sha: str) -> bytes:
    """Fetch a blob object by SHA and return its raw bytes."""
    url = f"{GITHUB_API}/{repo}/git/blobs/{sha}"
    response = session.get(url, timeout=(5, 30))
    response.raise_for_status()

    data = response.json()
    content = data.get("content", "")
    if data.get("encoding") == "base64":
        return base64.b64decode(content)
    return content.encode("utf-8", errors="ignore")


def fetch_content(
    session: requests.Session, repo: str, ref: str, path: str, sha: Optional[str]
) -> bytes:
    """Fetch file content by trying the raw URL first, then Git blobs API.

    First tries raw.githubusercontent.com (does not consume REST API rate).
    If that fails and a SHA is available, falls back to ``git/blobs``.
    """
    # Try raw.githubusercontent.com first (does not consume REST API rate limit).
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    r = session.get(raw_url, timeout=(5, 30))
    if r.status_code == 200:
        return r.content

    # Fallback only if we have a SHA (zipball-based trees don't).
    if not sha:
        raise requests.HTTPError(f"raw fetch failed and no SHA available for {path}")

    url = f"{GITHUB_API}/{repo}/git/blobs/{sha}"
    r = session.get(url, timeout=(5, 30))
    r.raise_for_status()
    j = r.json()
    if j.get("encoding") == "base64":
        return base64.b64decode(j.get("content", b""))
    return (j.get("content") or "").encode("utf-8", errors="ignore")


def get_tree_via_zipball(session, repo: str, ref: str):
    """Return a file list by downloading the GitHub zipball for a ref.

    Uses the zipball endpoint instead of the Trees REST API to avoid consuming
    the REST rate limit (aside from the single zipball request).
    """
    url = f"https://api.github.com/repos/{repo}/zipball/{ref}"
    response = session.get(url, timeout=(5, 60), allow_redirects=True)
    response.raise_for_status()

    data = io.BytesIO(response.content)
    files = []

    with zipfile.ZipFile(data) as zfile:
        names = zfile.namelist()
        if names and names[0].endswith('/'):
            root = names[0].split('/', 1)[0] + '/'
        else:
            root = ''

        for info in zfile.infolist():
            if info.is_dir():
                continue

            filename = info.filename
            if root and filename.startswith(root):
                path = filename[len(root):]
            else:
                path = filename

            if path:
                files.append({
                    "path": path,
                    "sha": None,
                    "size": info.file_size,
                })

    return files


def get_raw_tree_cached(session, repo: str, ref: str, token_present: bool):
    """Return a cached or freshly fetched repository file tree.

    Tries the primary REST-based ``get_tree`` first. If it raises a 403 and no
    token is present, falls back to :func:`get_tree_via_zipball`. Results are
    cached when no token is present.
    """
    key = cache_key("rawtree", repo=repo, ref=ref)
    cached = None if token_present else cache_get(key)
    if cached:
        return cached

    try:
        files = get_tree(session, repo, ref)  # primary attempt (REST)
    except requests.HTTPError as exc:
        # On 403 and without a token → use ZIP fallback
        if (
            exc.response is not None
            and exc.response.status_code == 403
            and not token_present
        ):
            files = get_tree_via_zipball(session, repo, ref)
        else:
            raise

    if not token_present:
        cache_set(key, files)

    return files


# ---------------------------- /tree ----------------------------
@app.get("/tree")
def tree_get() -> Tuple[Response, int]:
    """Endpoint to list files in a repository tree with optional filters."""
    auth_err = require_api_key()
    if auth_err:
        return auth_err

    repo_like = request.args.get("repo") or request.args.get("repo_url")
    ref = request.args.get("ref") or request.args.get("branch") or "HEAD"
    path_prefix = request.args.get("path", "")
    include = csv_patterns(request.args.get("include"))
    exclude = csv_patterns(request.args.get("exclude"))
    no_cache = request.args.get("no_cache", "0") == "1"
    token = request.args.get("token") or os.getenv("GITHUB_TOKEN")

    if not repo_like:
        return jsonify({"error": "repo is required"}), 400

    repo, _ = parse_repo_input(repo_like)
    sess = make_session(token)
    token_present = bool(token)

    key = cache_key(
        "tree",
        repo=repo,
        ref=ref,
        path=path_prefix,
        inc="|".join(include),
        exc="|".join(exclude),
    )

    payload: Optional[Dict[str, Any]] = None if (no_cache or token) else cache_get(key)

    if not payload:
        files = get_raw_tree_cached(sess, repo, ref, token_present)
        files = filter_files(files, include, exclude, path_prefix)
        payload = {
            "repo": repo,
            "ref": ref,
            "path": path_prefix,
            "files": files,
            "from_cache": False,
        }
        if not token and not no_cache:
            cache_set(key, payload)
    else:
        payload["from_cache"] = True

    app.logger.info(
        "Tree: %s ref=%s prefix=%s -> %d files",
        repo,
        ref,
        path_prefix,
        len(payload["files"]),
    )
    return jsonify(payload), 200


# ---------------------------- /fetch ----------------------------
@app.get("/fetch")
@app.post("/fetch")
def fetch_route() -> Tuple[Response, int]:
    """Endpoint to fetch file metadata and optionally contents from a repo."""
    auth_err = require_api_key()
    if auth_err:
        return auth_err

    if request.method == "POST":
        data = request.get_json(silent=True) or {}

        def get_param(key: str, default: Any = None) -> Any:
            """Return a value from the provided JSON body with a default."""
            return data.get(key, default)

    else:
        def get_param(key: str, default: Any = None) -> Any:
            """Return a value from the query string with a default."""
            return request.args.get(key, default)

    repo_like = get_param("repo") or get_param("repo_url")
    ref = get_param("ref") or get_param("branch") or "HEAD"
    path_prefix = get_param("path", "")
    include = csv_patterns(get_param("include"))
    exclude = csv_patterns(get_param("exclude"))
    include_binary = str(get_param("include_binary", "false")).lower() in ("1", "true", "yes")
    max_files = int(get_param("max_files", MAX_FILES_DEFAULT))
    max_bytes = int(get_param("max_bytes", MAX_BYTES_DEFAULT))
    no_cache = str(get_param("no_cache", "0")) == "1"
    token = get_param("token") or os.getenv("GITHUB_TOKEN")

    if not repo_like:
        return jsonify({"error": "repo is required"}), 400

    repo, _ = parse_repo_input(repo_like)
    sess = make_session(token)
    token_present = bool(token)

    # Use a single cache key covering filters and limits (for anonymous usage).
    key = cache_key(
        "fetch",
        repo=repo,
        ref=ref,
        path=path_prefix,
        inc="|".join(include),
        exc="|".join(exclude),
        ib=str(include_binary),
        mf=str(max_files),
        mb=str(max_bytes),
    )

    cached = None if (no_cache or token) else cache_get(key)

    if cached:
        cached["from_cache"] = True
        if isinstance(cached.get("stats"), dict):
            cached["stats"]["from_cache"] = True
        app.logger.info("Fetch (cache): %s ref=%s -> %d files", repo, ref, len(cached.get("files", [])))
        return jsonify(cached), 200

    # 1) Retrieve the tree and apply filters.
    tree = get_raw_tree_cached(sess, repo, ref, token_present)
    filtered = filter_files(tree, include, exclude, path_prefix)

    # 2) Apply limits progressively.
    files_out: List[Dict[str, Any]] = []
    total_bytes = 0

    for f in filtered:
        if len(files_out) >= max_files:
            break

        size = int(f.get("size") or 0)
        # Stop if adding this file would exceed the byte limit.
        if total_bytes + size > max_bytes:
            break

        path = f["path"]
        sha = f["sha"]
        mime = guess_mime(path)
        binary_flag = is_binary_mime(mime)

        if binary_flag and not include_binary:
            files_out.append(
                {
                    "path": path,
                    "size": size,
                    "sha": sha,
                    "is_binary": True,
                    "mime": mime,
                }
            )
            total_bytes += size
            continue

        # 3) Fetch content on demand (with graceful error handling).
        try:
            blob = fetch_content(sess, repo, ref, path, sha)
        except requests.HTTPError as exc:
            files_out.append({
                "path": path,
                "size": size,
                "sha": sha,
                "is_binary": binary_flag,
                "mime": mime,
                "error": f"content_unavailable: {exc}",
            })
            continue  # keep processing other files

        size = len(blob)
        if total_bytes + size > max_bytes:
            break

        if binary_flag and include_binary:
            payload: Dict[str, Any] = {
                "path": path,
                "size": size,
                "sha": sha,
                "is_binary": True,
                "mime": mime,
                "content_base64": base64.b64encode(blob).decode("ascii"),
            }
        else:
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                text = blob.decode("latin-1", errors="replace")
            payload = {
                "path": path,
                "size": size,
                "sha": sha,
                "is_binary": False,
                "mime": mime,
                "content": text,
            }

        files_out.append(payload)
        total_bytes += size

    result: Dict[str, Any] = {
        "repo": repo,
        "ref": ref,
        "path": path_prefix,
        "limits": {"max_files": max_files, "max_bytes": max_bytes},
        "from_cache": False,
        "stats": {
            "files": len(files_out),
            "bytes": total_bytes,
            "from_cache": False,
        },
        "files": files_out,
    }

    if not token and not no_cache:
        cache_set(key, result)

    app.logger.info(
        "Fetch: %s ref=%s prefix=%s -> %d files (%d bytes)",
        repo,
        ref,
        path_prefix,
        len(files_out),
        total_bytes,
    )
    return jsonify(result), 200


# ---------------------------- Error handling ----------------------------
@app.errorhandler(Exception)
def _handle_unexpected(e: Exception):
    app.logger.exception("Unhandled error: %s", e)
    return jsonify({"error": "internal_error"}), 500


# ---------------------------- Run (dev) ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5555, debug=False)
