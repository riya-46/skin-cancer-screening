import hashlib
import os
import re
import time
import zipfile
from html import escape
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_PRIORITY = ["benign", "malignant", "invalid", "unknown", "not_skin_lesion"]
SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}
DEFAULT_LIBRARY_ROOT = Path("sample_images")
CACHE_ROOT = Path(".cache/demo_library")
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        secret_value = st.secrets[name]
    except Exception:
        return default

    if secret_value is None:
        return default
    return str(secret_value).strip() or default


def inspect_library_config() -> dict[str, str]:
    local_dir = get_setting("DEMO_LIBRARY_DIR")
    drive_folder_id = get_setting("DEMO_LIBRARY_DRIVE_FOLDER_ID")
    drive_folder_url = get_setting("DEMO_LIBRARY_DRIVE_FOLDER_URL")
    drive_file_id = get_setting("DEMO_LIBRARY_DRIVE_FILE_ID")
    drive_url = get_setting("DEMO_LIBRARY_DRIVE_URL")
    label = get_setting("DEMO_LIBRARY_LABEL")

    if local_dir:
        return {
            "mode": "directory",
            "label": label or "Connected dataset folder",
            "directory": local_dir,
            "drive_folder_id": "",
            "drive_folder_url": "",
            "drive_file_id": "",
            "drive_url": "",
        }

    if drive_folder_id or drive_folder_url:
        return {
            "mode": "drive_folder",
            "label": label or "Google Drive folder dataset",
            "directory": "",
            "drive_folder_id": drive_folder_id,
            "drive_folder_url": drive_folder_url,
            "drive_file_id": "",
            "drive_url": "",
        }

    if drive_file_id or drive_url:
        return {
            "mode": "drive",
            "label": label or "Google Drive dataset",
            "directory": "",
            "drive_folder_id": "",
            "drive_folder_url": "",
            "drive_file_id": drive_file_id,
            "drive_url": drive_url,
        }

    return {
        "mode": "bundled",
        "label": label or "Bundled image library",
        "directory": str(DEFAULT_LIBRARY_ROOT),
        "drive_folder_id": "",
        "drive_folder_url": "",
        "drive_file_id": "",
        "drive_url": "",
    }


def resolve_backend_url() -> str:
    backend_url = get_setting("BACKEND_URL")
    if backend_url:
        return backend_url

    backend_hostport = get_setting("BACKEND_HOSTPORT")
    if backend_hostport:
        return f"http://{backend_hostport}"

    return DEFAULT_BACKEND_URL


def extract_drive_file_id(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""

    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", candidate):
        return candidate

    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"/d/([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, candidate)
        if match:
            return match.group(1)

    return ""


def extract_drive_folder_id(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""

    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", candidate):
        return candidate

    patterns = [
        r"/folders/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, candidate)
        if match:
            return match.group(1)

    return ""


def is_drive_url(value: str) -> bool:
    return "drive.google.com" in value.lower()


def is_drive_folder_url(value: str) -> bool:
    lower_value = value.lower()
    return "drive.google.com" in lower_value and "/folders/" in lower_value


def get_confirm_token(response: requests.Response) -> str:
    for cookie_name, cookie_value in response.cookies.items():
        if cookie_name.startswith("download_warning"):
            return cookie_value

    if "text/html" not in response.headers.get("Content-Type", "").lower():
        return ""

    html = response.text
    patterns = [
        r'name="confirm" value="([^"]+)"',
        r"confirm=([0-9A-Za-z_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1)

    return ""


def stream_to_file(response: requests.Response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                output_file.write(chunk)


def download_google_drive_file(source_value: str, destination: Path) -> None:
    file_id = extract_drive_file_id(source_value)
    if not file_id:
        raise RuntimeError(
            "Google Drive file id detect nahi hua. Public share link ya file id set karo."
        )

    session = requests.Session()
    download_url = "https://drive.google.com/uc?export=download"
    response = session.get(
        download_url,
        params={"id": file_id},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()

    token = get_confirm_token(response)
    if token:
        response.close()
        response = session.get(
            download_url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=180,
        )
        response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    content_disposition = response.headers.get("Content-Disposition", "").lower()
    if "text/html" in content_type and ".zip" not in content_disposition:
        response.close()
        raise RuntimeError(
            "Google Drive se zip download nahi hui. File ko 'Anyone with the link' access do."
        )

    stream_to_file(response, destination)
    response.close()


def download_remote_file(source_url: str, destination: Path) -> None:
    response = requests.get(source_url, stream=True, timeout=180)
    response.raise_for_status()
    stream_to_file(response, destination)
    response.close()


def ensure_archive_is_zip(archive_path: Path) -> None:
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(
            "Configured dataset archive zip format me nahi hai. Google Drive par zip upload karo."
        )


def prepare_drive_library(drive_file_id: str, drive_url: str) -> str:
    source_key = drive_file_id or drive_url
    cache_key = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    source_root = CACHE_ROOT / cache_key
    archive_path = source_root / "library.zip"
    extract_root = source_root / "extracted"
    ready_marker = source_root / ".ready"

    source_root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        if drive_file_id:
            download_google_drive_file(drive_file_id, archive_path)
        else:
            if is_drive_url(drive_url):
                download_google_drive_file(drive_url, archive_path)
            else:
                download_remote_file(drive_url, archive_path)

    ensure_archive_is_zip(archive_path)

    if not ready_marker.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
        ready_marker.write_text("ready", encoding="utf-8")

    return str(extract_root.resolve())


def count_supported_images(root: Path) -> int:
    if not root.exists():
        return 0

    return sum(
        1
        for file_path in root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_drive_folder_library(
    drive_folder_id: str,
    drive_folder_url: str,
) -> dict[str, str]:
    source_key = drive_folder_id or drive_folder_url
    cache_key = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    source_root = CACHE_ROOT / f"folder_{cache_key}"
    extract_root = source_root / "extracted"
    ready_marker = source_root / ".ready"
    partial_marker = source_root / ".partial_warning"

    if ready_marker.exists():
        return {"root": str(extract_root.resolve()), "warning": ""}

    if partial_marker.exists() and count_supported_images(extract_root) > 0:
        return {
            "root": str(extract_root.resolve()),
            "warning": partial_marker.read_text(encoding="utf-8").strip(),
        }

    source_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    folder_id = extract_drive_folder_id(source_key)
    if drive_folder_url:
        folder_url = drive_folder_url.strip()
    elif folder_id:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    else:
        raise RuntimeError(
            "Google Drive folder id detect nahi hua. Public folder link ya folder id set karo."
        )

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive folder support ke liye `gdown` install hona chahiye."
        ) from exc

    download_kwargs = {
        "output": str(extract_root),
        "quiet": True,
    }
    if folder_id:
        folder_items = gdown.download_folder(
            id=folder_id,
            skip_download=True,
            **download_kwargs,
        )
    else:
        folder_items = gdown.download_folder(
            url=folder_url,
            skip_download=True,
            **download_kwargs,
        )

    if not folder_items:
        raise RuntimeError(
            "Google Drive folder se files load nahi hui. Folder ko 'Anyone with the link' access do."
        )

    skipped_files: list[str] = []

    for folder_item in folder_items:
        local_path = Path(folder_item.local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.suffix:
            download_output = str(local_path)
        else:
            download_output = str(local_path.parent) + os.sep

        try:
            gdown.download(
                url=f"https://drive.google.com/uc?id={folder_item.id}",
                output=download_output,
                quiet=True,
                resume=True,
            )
        except Exception:
            skipped_files.append(folder_item.path)

    image_count = count_supported_images(extract_root)
    if image_count == 0:
        raise RuntimeError(
            "Google Drive folder se image files load nahi hui. Folder ko 'Anyone with the link' access do ya dataset zip/local folder use karo."
        )

    if skipped_files:
        warning = (
            f"Google Drive folder partially load hua. {len(skipped_files)} file(s) access nahi hui, "
            "isliye abhi available images hi dikhayi ja rahi hain."
        )
        partial_marker.write_text(warning, encoding="utf-8")
        return {"root": str(extract_root.resolve()), "warning": warning}

    if partial_marker.exists():
        partial_marker.unlink()
    ready_marker.write_text("ready", encoding="utf-8")
    return {"root": str(extract_root.resolve()), "warning": ""}


@st.cache_resource(show_spinner=False)
def resolve_library_root(config: tuple[str, str, str, str, str, str]) -> dict[str, str]:
    mode, directory, drive_folder_id, drive_folder_url, drive_file_id, drive_url = config

    if mode == "directory":
        dataset_root = Path(directory).expanduser()
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Configured dataset folder nahi mila: {dataset_root}"
            )
        return {"root": str(dataset_root.resolve()), "mode": mode}

    if mode == "drive_folder":
        drive_folder_library = prepare_drive_folder_library(
            drive_folder_id,
            drive_folder_url,
        )
        return {
            "root": drive_folder_library["root"],
            "mode": mode,
            "warning": drive_folder_library["warning"],
        }

    if mode == "drive":
        return {
            "root": prepare_drive_library(drive_file_id, drive_url),
            "mode": mode,
        }

    dataset_root = DEFAULT_LIBRARY_ROOT
    if not dataset_root.exists():
        raise FileNotFoundError(
            "Bundled image library missing hai. `sample_images/` ya external dataset configure karo."
        )
    return {"root": str(dataset_root.resolve()), "mode": mode}


def infer_split_name(relative_path: Path) -> str:
    for part in relative_path.parts[:-1]:
        normalized = part.lower()
        if normalized in SPLIT_ALIASES:
            return SPLIT_ALIASES[normalized]
    return "library"


def infer_class_name(relative_path: Path) -> str:
    folders = [part.lower() for part in relative_path.parts[:-1]]

    for candidate in CLASS_PRIORITY:
        if candidate in folders:
            return candidate

    for part in reversed(folders):
        if part not in SPLIT_ALIASES and part not in {"images", "image", "dataset", "archive"}:
            return part

    return "unlabeled"


def prettify_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


@st.cache_data(show_spinner=False)
def build_library_index(root_str: str) -> list[dict[str, str]]:
    dataset_root = Path(root_str)
    entries: list[dict[str, str]] = []

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(dataset_root)
        class_name = infer_class_name(relative_path)
        split_name = infer_split_name(relative_path)
        stable_key = hashlib.md5(str(relative_path).encode("utf-8")).hexdigest()[:12]

        entries.append(
            {
                "key": stable_key,
                "path": str(file_path.resolve()),
                "filename": file_path.name,
                "relative_path": str(relative_path).replace("\\", "/"),
                "class_name": class_name,
                "class_label": prettify_label(class_name),
                "split_name": split_name,
                "split_label": prettify_label(split_name),
            }
        )

    class_order = {name: index for index, name in enumerate(CLASS_PRIORITY)}
    entries.sort(
        key=lambda item: (
            class_order.get(item["class_name"], 99),
            item["split_name"],
            item["filename"].lower(),
        )
    )
    return entries


@st.cache_data(show_spinner=False)
def load_gallery_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as source_image:
        return source_image.convert("RGB").copy()


@st.cache_data(show_spinner=False)
def load_gallery_thumbnail(image_path: str) -> Image.Image:
    with Image.open(image_path) as source_image:
        fitted_image = ImageOps.fit(
            source_image.convert("RGB"),
            (360, 240),
            method=Image.Resampling.LANCZOS,
        )
        return fitted_image.copy()


def read_gallery_image_bytes(image_path: str) -> bytes:
    return Path(image_path).read_bytes()


def format_percent(probability: float) -> str:
    percentage = probability * 100
    if percentage >= 99.995:
        return "99.99%+"
    if percentage <= 0.005:
        return "<0.01%"
    return f"{percentage:.2f}%"


def scroll_to_anchor(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
        const scrollToAnchor = () => {{
            const anchor = window.parent.document.getElementById("{anchor_id}");
            if (anchor) {{
                anchor.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }};
        window.parent.requestAnimationFrame(scrollToAnchor);
        </script>
        """,
        height=0,
    )


def get_risk_style(risk_level: str) -> tuple[str, str]:
    mapping = {
        "Low Risk": ("#0f766e", "rgba(15, 118, 110, 0.12)"),
        "Suspicious": ("#b45309", "rgba(180, 83, 9, 0.14)"),
        "High Risk": ("#b91c1c", "rgba(185, 28, 28, 0.14)"),
        "Invalid Image": ("#475569", "rgba(71, 85, 105, 0.14)"),
    }
    return mapping.get(risk_level, ("#1f2937", "rgba(31, 41, 55, 0.12)"))


def get_library_summary(entries: list[dict[str, str]]) -> dict[str, int]:
    class_count = len({entry["class_name"] for entry in entries})
    split_count = len({entry["split_name"] for entry in entries})
    return {
        "image_count": len(entries),
        "class_count": class_count,
        "split_count": split_count,
    }


def init_session_state() -> None:
    defaults = {
        "prediction_result": None,
        "error_message": None,
        "show_result": False,
        "scroll_to_analyze": False,
        "scroll_to_result": False,
        "selected_library_path": "",
        "uploader_key": 0,
        "library_page": 1,
        "last_upload_signature": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workspace() -> None:
    st.session_state.prediction_result = None
    st.session_state.error_message = None
    st.session_state.show_result = False
    st.session_state.scroll_to_analyze = False
    st.session_state.scroll_to_result = False
    st.session_state.selected_library_path = ""
    st.session_state.last_upload_signature = ""
    st.session_state.uploader_key += 1
    st.session_state.library_page = 1


def track_uploaded_file(uploaded_file) -> None:
    if uploaded_file is None:
        st.session_state.last_upload_signature = ""
        st.session_state.scroll_to_analyze = False
        return

    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if signature != st.session_state.last_upload_signature:
        st.session_state.prediction_result = None
        st.session_state.error_message = None
        st.session_state.show_result = False
        st.session_state.scroll_to_analyze = True
        st.session_state.scroll_to_result = False
    st.session_state.last_upload_signature = signature


def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --page-bg: linear-gradient(180deg, #edf2ff 0%, #e8eefb 100%);
            --card-bg: rgba(255, 255, 255, 0.96);
            --card-muted-bg: #f7faff;
            --card-border: #d9e1f1;
            --card-shadow: 0 18px 42px rgba(76, 93, 129, 0.14);
            --ink-strong: #13294b;
            --ink: #40597c;
            --muted: #7b89a6;
            --accent: #4f46e5;
            --accent-strong: #4338ca;
            --accent-soft: rgba(79, 70, 229, 0.10);
            --soft-fill: #f9fbff;
            --soft-line: #cfd8ea;
            --subtle-line: #e3e9f5;
            --progress-track: #e6ebf7;
            --footer-ink: #495f83;
            --disclaimer-ink: #c72626;
            --disclaimer-bg: rgba(255, 244, 244, 0.88);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --page-bg: linear-gradient(180deg, #0b1323 0%, #0f1b30 100%);
                --card-bg: rgba(18, 28, 48, 0.94);
                --card-muted-bg: #12203a;
                --card-border: #243556;
                --card-shadow: 0 18px 48px rgba(0, 0, 0, 0.38);
                --ink-strong: #f4f7ff;
                --ink: #d4ddf4;
                --muted: #99a8c7;
                --accent: #8d8aff;
                --accent-strong: #a7a4ff;
                --accent-soft: rgba(141, 138, 255, 0.16);
                --soft-fill: #101a2e;
                --soft-line: #334666;
                --subtle-line: #243556;
                --progress-track: #27354d;
                --footer-ink: #b9c7e2;
                --disclaimer-ink: #ff9b9b;
                --disclaimer-bg: rgba(70, 24, 29, 0.50);
            }
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", "Segoe UI", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background: var(--page-bg);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        .app-hero {
            text-align: center;
            padding: 0.3rem 0 1.65rem;
        }

        .hero-lockup {
            display: inline-flex;
            align-items: center;
            gap: 0.9rem;
            color: var(--ink-strong);
        }

        .hero-icon {
            width: 52px;
            height: 52px;
            color: var(--accent);
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .hero-title {
            font-size: clamp(2.35rem, 4vw, 3.5rem);
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--ink-strong);
            margin: 0;
        }

        .hero-subtitle {
            margin-top: 1rem;
            font-size: 1.08rem;
            color: var(--ink);
        }

        .section-card-title {
            color: var(--ink-strong);
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.32rem;
            letter-spacing: -0.02em;
            text-align: center;
        }

        .section-card-copy {
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.75;
            text-align: center;
            margin-bottom: 1.15rem;
        }

        .upload-section-intro {
            padding-top: 0.95rem;
            padding-bottom: 0.55rem;
        }

        .layout-gap {
            height: 2.7rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--card-border) !important;
            border-radius: 26px !important;
            background: var(--card-bg) !important;
            box-shadow: var(--card-shadow) !important;
            padding: 0.55rem !important;
        }

        .panel-title {
            color: var(--ink-strong);
            font-size: 1.28rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            text-align: center;
            letter-spacing: -0.02em;
        }

        .panel-copy {
            color: var(--ink);
            font-size: 0.96rem;
            line-height: 1.75;
            text-align: center;
            margin-bottom: 1.15rem;
        }

        div[data-testid="stRadio"] {
            border-bottom: 1px solid var(--subtle-line);
            margin-bottom: 1.35rem;
            padding-bottom: 0.08rem;
        }

        div[data-testid="stRadio"] > div {
            gap: 2rem;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent;
            border-radius: 0 !important;
            padding: 0 0 0.95rem !important;
            margin-right: 0 !important;
            min-height: auto !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"] p {
            color: var(--muted) !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            border-bottom-color: var(--accent) !important;
        }

        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
            color: var(--accent) !important;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: var(--soft-fill) !important;
            border: 2px dashed var(--soft-line) !important;
            border-radius: 24px !important;
            min-height: 280px;
            padding: 2rem 1.35rem !important;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] > div {
            color: var(--ink-strong) !important;
            font-size: 1.08rem !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--muted) !important;
            font-size: 0.96rem !important;
        }

        div[data-testid="stFileUploaderDropzone"] svg {
            color: var(--muted) !important;
            fill: var(--muted) !important;
        }

        div[data-testid="stFileUploaderFile"] {
            background: var(--card-muted-bg) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 18px !important;
            margin-top: 0.8rem;
            padding: 0.15rem 0.1rem;
        }

        div[data-testid="stFileUploaderFileName"] {
            color: var(--ink-strong) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFileUploaderFile"] small {
            color: var(--muted) !important;
            font-weight: 600 !important;
        }

        div[data-testid="stFileUploaderFile"] svg,
        div[data-testid="stFileUploaderDeleteBtn"] button {
            color: var(--ink-strong) !important;
            fill: var(--ink-strong) !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 14px;
            min-height: 46px;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: none;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
            color: #ffffff;
            border: 1px solid transparent;
        }

        .stButton > button[kind="secondary"] {
            background: var(--card-muted-bg);
            color: var(--ink-strong);
            border: 1px solid var(--card-border);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {
            border-radius: 14px !important;
            border: 1px solid var(--card-border) !important;
            background: var(--soft-fill) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input {
            color: var(--ink-strong) !important;
        }

        div[role="listbox"] {
            background: var(--card-bg) !important;
            border: 1px solid var(--card-border) !important;
            color: var(--ink-strong) !important;
        }

        div[role="option"] {
            color: var(--ink-strong) !important;
        }

        div[role="option"][aria-selected="true"] {
            background: var(--accent-soft) !important;
        }

        div[data-testid="stImage"] img {
            border-radius: 20px;
            border: 1px solid var(--card-border);
        }

        .sample-caption,
        div[data-testid="stCaptionContainer"] p,
        div[data-testid="stFileUploader"] > label p,
        div[data-testid="stSelectbox"] > label p {
            color: var(--muted) !important;
            font-weight: 600 !important;
        }

        .sample-gallery-title {
            color: var(--ink-strong);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .sample-gallery-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 0.9rem;
        }

        .sample-card-title {
            color: var(--ink-strong);
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 0.6rem;
            line-height: 1.4;
        }

        .sample-card-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 0.18rem;
            min-height: 2.35rem;
        }

        .sample-selected-pill {
            display: inline-flex;
            align-items: center;
            margin-top: 0.55rem;
            margin-bottom: 0.55rem;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
        }

        .selected-preview-title {
            color: var(--ink-strong);
            font-size: 1rem;
            font-weight: 700;
            margin-top: 1.55rem;
            margin-bottom: 0.95rem;
            text-align: center;
        }

        .preview-meta {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.6;
            text-align: center;
            margin-top: 0.85rem;
        }

        .results-empty {
            min-height: 250px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: var(--muted);
            gap: 1rem;
        }

        .results-empty svg {
            width: 68px;
            height: 68px;
            color: var(--muted);
        }

        .results-empty-copy {
            font-size: 1rem;
            color: var(--muted);
        }

        .result-card {
            background: var(--card-muted-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.36rem 0.75rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
        }

        .result-headline {
            color: var(--ink-strong);
            font-size: clamp(1.75rem, 3.4vw, 2.4rem);
            line-height: 1.1;
            font-weight: 800;
            margin: 0.7rem 0 0.35rem;
        }

        .result-copy {
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.7;
        }

        .metric-card {
            background: var(--card-muted-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1rem;
            min-height: 132px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .metric-value {
            color: var(--ink-strong);
            font-size: 1.7rem;
            line-height: 1;
            font-weight: 800;
        }

        .metric-caption {
            color: var(--ink);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.65rem;
        }

        div[data-testid="stProgressBar"] > div {
            background: var(--progress-track);
        }

        div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(135deg, var(--accent) 0%, #6c8cff 100%);
        }

        .recommendation-box {
            margin-top: 1rem;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: var(--card-muted-bg);
            border: 1px solid var(--card-border);
            color: var(--ink);
            line-height: 1.7;
        }

        .disclaimer-box {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            color: var(--disclaimer-ink);
            background: var(--disclaimer-bg);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }

        .disclaimer-icon {
            width: 34px;
            height: 34px;
            flex: 0 0 auto;
        }

        .disclaimer-title {
            color: var(--disclaimer-ink);
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        .disclaimer-copy {
            color: var(--disclaimer-ink);
            font-size: 1rem;
            line-height: 1.75;
        }

        .app-footer {
            text-align: center;
            padding: 1.8rem 0 0.3rem;
            color: var(--footer-ink);
            font-size: 0.98rem;
            line-height: 1.8;
        }

        div[data-testid="stAlert"] {
            border-radius: 16px;
            background: var(--card-muted-bg) !important;
            border: 1px solid var(--card-border) !important;
        }

        div[data-testid="stAlertContainer"] {
            background: transparent !important;
            color: var(--ink-strong) !important;
        }

        div[data-testid^="stAlertContent"] p,
        div[data-testid^="stAlertContent"] li,
        div[data-testid^="stAlertContent"] span,
        div[data-testid^="stAlertContent"] div {
            color: var(--ink-strong) !important;
        }

        div[data-testid="stAlert"] svg {
            color: var(--ink-strong) !important;
            fill: var(--ink-strong) !important;
        }

        div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
            border-left: 4px solid #34c37b !important;
        }

        div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
            border-left: 4px solid #d7a11d !important;
        }

        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
            border-left: 4px solid #df5b5b !important;
        }

        div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
            border-left: 4px solid var(--accent) !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.7rem;
            }

            .hero-lockup {
                gap: 0.6rem;
            }

            .app-hero {
                padding: 0.05rem 0 0.85rem;
            }

            div[data-testid="stFileUploaderDropzone"] {
                min-height: 150px;
                padding: 0.85rem 0.95rem !important;
                border-radius: 18px !important;
            }

            div[data-testid="stFileUploaderDropzoneInstructions"] > div {
                font-size: 0.88rem !important;
            }

            div[data-testid="stFileUploaderDropzoneInstructions"] small {
                font-size: 0.78rem !important;
                line-height: 1.35 !important;
            }

            .stButton > button {
                min-height: 40px;
            }

            .results-empty {
                min-height: 210px;
            }

            .layout-gap {
                height: 2rem;
            }

            .disclaimer-box {
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-lockup">
                <div class="hero-icon">
                    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
                        <path d="M5 34H18L25 15L34 49L42 28H59" stroke="currentColor" stroke-width="4.8" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="hero-title">Skin Cancer Screening System</div>
            </div>
            <div class="hero-subtitle">
                AI-assisted skin lesion analysis using DenseNet121
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_empty() -> None:
    st.markdown(
        """
        <div class="results-empty">
            <svg viewBox="0 0 64 64" fill="none" aria-hidden="true">
                <path d="M7 35H19L26 16L34 50L42 29H57" stroke="currentColor" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <div class="results-empty-copy">Upload and analyze an image to see results</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer-box">
            <svg class="disclaimer-icon" viewBox="0 0 32 32" fill="none" aria-hidden="true">
                <circle cx="16" cy="16" r="13" stroke="currentColor" stroke-width="2.4"/>
                <path d="M16 9V17" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
                <circle cx="16" cy="22.5" r="1.5" fill="currentColor"/>
            </svg>
            <div>
                <div class="disclaimer-title">Medical Disclaimer</div>
                <div class="disclaimer-copy">
                    This tool is for educational and screening purposes only. It is not a confirmed medical diagnosis.
                    Always consult with a qualified healthcare professional for proper medical advice and diagnosis.
                    This system is designed for close-up skin lesion images and may reject unsuitable uploads.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            Powered by DenseNet121 • PyTorch • FastAPI<br>
            Supports: Benign | Malignant | Invalid classifications
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sample_gallery(entries: list[dict[str, str]], source_label: str) -> dict[str, str] | None:
    st.markdown(
        f"""
        <div class="sample-gallery-title">{escape(source_label)}</div>
        <div class="sample-gallery-copy">
            Sample preview gallery se image choose karo. Select karte hi neeche larger preview dikh jayega.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("First row visible rahegi. Aur samples isi block ke andar scroll me milenge.")

    selected_path = st.session_state.selected_library_path
    selected_entry = next(
        (entry for entry in entries if entry["path"] == selected_path),
        None,
    )

    with st.container(height=385, border=False):
        grid_columns = st.columns(3, gap="small")
        for index, entry in enumerate(entries):
            with grid_columns[index % 3]:
                with st.container(border=True):
                    try:
                        st.image(
                            load_gallery_thumbnail(entry["path"]),
                            use_container_width=True,
                        )
                    except (FileNotFoundError, UnidentifiedImageError):
                        st.warning("Preview unavailable")
                        continue

                    st.markdown(
                        f"""
                        <div class="sample-card-title">{escape(entry["class_label"])}</div>
                        <div class="sample-card-meta">{escape(entry["relative_path"])}</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    is_selected = selected_path == entry["path"]
                    if is_selected:
                        st.markdown(
                            '<div class="sample-selected-pill">Selected</div>',
                            unsafe_allow_html=True,
                        )

                    if st.button(
                        "Use This Sample" if not is_selected else "Selected",
                        key=f"sample_pick_{entry['key']}",
                        use_container_width=True,
                        disabled=is_selected,
                    ):
                        st.session_state.selected_library_path = entry["path"]
                        st.session_state.prediction_result = None
                        st.session_state.error_message = None
                        st.session_state.show_result = False
                        st.session_state.scroll_to_analyze = True
                        st.session_state.scroll_to_result = False
                        selected_entry = entry
                        selected_path = entry["path"]

    if selected_entry is None and selected_path:
        selected_entry = next(
            (entry for entry in entries if entry["path"] == selected_path),
            None,
        )

    return selected_entry


def analyze_image(
    image_name: str,
    image_bytes: bytes,
    image_mime: str,
    backend_url: str,
) -> None:
    st.session_state.error_message = None
    st.session_state.prediction_result = None
    st.session_state.show_result = False

    status_box = st.empty()
    progress_box = st.empty()

    steps = [
        "Preparing image payload...",
        "Checking lesion framing and quality...",
        "Running model inference...",
        "Compiling risk summary...",
    ]

    try:
        for index, step in enumerate(steps, start=1):
            status_box.info(step)
            progress_box.progress(index * 20)
            time.sleep(0.24)

        response = requests.post(
            f"{backend_url}/predict",
            files={"file": (image_name, image_bytes, image_mime)},
            timeout=180,
        )
        progress_box.progress(100)

        if response.status_code == 200:
            st.session_state.prediction_result = response.json()
            st.session_state.show_result = True
            st.session_state.scroll_to_result = True
            status_box.success("Screening report ready.")
            return

        st.session_state.error_message = f"API Error: {response.status_code}"
    except requests.exceptions.Timeout:
        st.session_state.error_message = (
            "Backend response timeout hua. Free hosting par pehla request cold start ki wajah se slow ho sakta hai."
        )
    except requests.exceptions.ConnectionError:
        st.session_state.error_message = (
            "Backend connect nahi hua. FastAPI server run ho ya deployed URL reachable ho, yeh verify karo."
        )
    except Exception as exc:
        st.session_state.error_message = f"Unexpected error: {exc}"

    st.session_state.show_result = False
    st.session_state.scroll_to_result = True


def render_result_panel(prediction_result: dict[str, object] | None) -> None:
    if not prediction_result:
        render_results_empty()
        return

    predicted_class = str(prediction_result["predicted_class"])
    confidence = float(prediction_result["predicted_probability"])
    benign_probability = float(prediction_result["benign_probability"])
    malignant_probability = float(prediction_result["malignant_probability"])
    invalid_probability = float(prediction_result.get("invalid_probability", 0.0))
    risk_level = str(prediction_result["risk_level"])
    recommendation = str(prediction_result["recommendation"])
    is_valid_image = bool(prediction_result.get("is_valid_image", True))
    is_uncertain = bool(prediction_result.get("is_uncertain", False))

    risk_color, risk_background = get_risk_style(risk_level)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="status-pill" style="color:{risk_color}; background:{risk_background}; border:1px solid {risk_color};">
                {escape(risk_level)}
            </div>
            <div class="result-headline">{escape(predicted_class)}</div>
            <div class="result-copy">
                Confidence {escape(format_percent(confidence))}. {escape(recommendation)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3, gap="large")
    with metric_columns[0]:
        render_metric_card(
            "Benign Probability",
            format_percent(benign_probability),
            "Lower values are expected when malignant or invalid evidence dominates.",
        )
        st.progress(benign_probability)
    with metric_columns[1]:
        render_metric_card(
            "Malignant Probability",
            format_percent(malignant_probability),
            "This drives the risk flag when the threshold is crossed.",
        )
        st.progress(malignant_probability)
    with metric_columns[2]:
        render_metric_card(
            "Invalid Probability",
            format_percent(invalid_probability),
            "High values usually mean the upload is not a lesion-focused close-up.",
        )
        st.progress(invalid_probability)

    if not is_valid_image:
        st.warning(
            "Image lesion-focused nahi lag rahi. Close-up dermoscopic ya lesion-centered image upload karna better rahega."
        )

    if is_uncertain:
        st.info(
            "Model uncertain hai. Clearer crop, sharper focus, aur closer lesion framing se result better ho sakta hai."
        )

    st.markdown(
        f"""
        <div class="recommendation-box">
            <strong>Recommendation:</strong> {escape(recommendation)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Skin Cancer Screening",
        page_icon=":microscope:",
        layout="wide",
    )

    init_session_state()
    inject_styles()

    backend_url = resolve_backend_url()

    render_hero()

    sample_entries: list[dict[str, str]] = []
    sample_error = ""
    try:
        sample_root = resolve_library_root(("bundled", "", "", "", "", ""))
        sample_entries = build_library_index(sample_root["root"])
        if not sample_entries:
            sample_error = "`sample_images/` folder me supported image files nahi mili."
    except Exception as exc:
        sample_error = str(exc)

    selected_image = None
    image_name = ""
    image_bytes = b""
    image_mime = "image/jpeg"
    preview_meta = ""
    run_scan = False

    center_column = st.columns([1.1, 4.8, 1.1], gap="large")[1]

    with center_column:
        with st.container(border=True):
            st.markdown(
                """
                <div class="upload-section-intro">
                    <div class="section-card-title">Upload Section</div>
                    <div class="section-card-copy">
                        Upload a lesion image, ya bundled sample gallery me se ek image choose karo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            source_mode = st.radio(
                "Input Source",
                options=["Upload Image", "Sample Images"],
                horizontal=True,
                label_visibility="collapsed",
            )

            uploaded_file = None
            selected_entry = None

            if source_mode == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload your image",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"uploader_{st.session_state.uploader_key}",
                    label_visibility="collapsed",
                )
                st.caption("PNG, JPG, JPEG, WEBP up to 10MB")
                track_uploaded_file(uploaded_file)

                if uploaded_file is not None:
                    image_bytes = uploaded_file.getvalue()
                    image_name = uploaded_file.name
                    image_mime = uploaded_file.type or image_mime
                    try:
                        selected_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        preview_meta = image_name
                    except UnidentifiedImageError:
                        st.session_state.error_message = "Uploaded image open nahi hui. Valid image file choose karo."
                        selected_image = None
                        st.session_state.scroll_to_analyze = False
                        image_bytes = b""
            else:
                track_uploaded_file(None)
                if sample_error:
                    st.warning(sample_error)
                else:
                    selected_entry = render_sample_gallery(
                        sample_entries,
                        "Sample Images",
                    )

                if selected_entry is not None:
                    try:
                        image_bytes = read_gallery_image_bytes(selected_entry["path"])
                        image_name = selected_entry["filename"]
                        selected_image = load_gallery_image(selected_entry["path"])
                        preview_meta = (
                            f"{selected_entry['class_label']} • {selected_entry['split_label']} • "
                            f"{selected_entry['relative_path']}"
                        )
                    except (FileNotFoundError, UnidentifiedImageError):
                        st.session_state.selected_library_path = ""
                        st.session_state.scroll_to_analyze = False
                        st.warning("Selected sample image load nahi hui. Dusri image choose karo.")

            if selected_image is not None:
                st.markdown('<div id="analyze-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="selected-preview-title">Selected Preview</div>',
                    unsafe_allow_html=True,
                )
                preview_columns = st.columns([1.45, 2.7, 1.45])
                with preview_columns[1]:
                    st.image(selected_image, use_container_width=True)
                    if preview_meta:
                        st.markdown(
                            f'<div class="preview-meta">{escape(preview_meta)}</div>',
                            unsafe_allow_html=True,
                        )

            button_columns = st.columns(2, gap="small")
            with button_columns[0]:
                run_scan = st.button(
                    "Analyze Image",
                    type="primary",
                    disabled=selected_image is None or not image_bytes,
                    use_container_width=True,
                )
            with button_columns[1]:
                if st.button(
                    "Reset",
                    type="secondary",
                    use_container_width=True,
                ):
                    reset_workspace()
                    st.rerun()

            st.caption("First request may take a little longer while the backend wakes up.")

            if run_scan and image_bytes:
                st.session_state.scroll_to_analyze = False
                analyze_image(image_name, image_bytes, image_mime, backend_url)

            if st.session_state.scroll_to_analyze and selected_image is not None:
                scroll_to_anchor("analyze-anchor")
                st.session_state.scroll_to_analyze = False

        st.markdown('<div class="layout-gap"></div>', unsafe_allow_html=True)
        st.markdown('<div id="analysis-results-anchor"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-title">Analysis Results</div>
                <div class="panel-copy">
                    Result yahan center me render hoga. Analyze complete hote hi page automatically yahan scroll karega.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.error_message:
                st.warning(st.session_state.error_message)
            render_result_panel(st.session_state.prediction_result)

        if st.session_state.scroll_to_result:
            scroll_to_anchor("analysis-results-anchor")
            st.session_state.scroll_to_result = False

        st.markdown('<div class="layout-gap"></div>', unsafe_allow_html=True)

        with st.container(border=True):
            render_disclaimer()

        render_footer()


if __name__ == "__main__":
    main()
