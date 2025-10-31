from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class Config:
    MORPHEUS_API_URL: str
    MORPHEUS_API_KEY: Optional[str]
    CORE_API_URL: str
    CORE_API_KEY: Optional[str]
    N8N_API_KEY: Optional[str]
    PERCEPTION_WEBHOOK_URL: Optional[str]
    LOG_FILE: str
    RESULTS_BASE_DIR: str
    RESULTS_RAW_DIR: str
    KNOWLEDGE_BASE_PATH: str
    ASSURANCE_REPORTS_DIR: str
    TMP_DIR: str
    SWARM_MAX_WORKERS: int
    DRY_RUN: bool


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _try_parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_env_files() -> None:
    """Load environment variables from supported .env files if present."""
    candidates: Iterable[Optional[str]] = (
        os.getenv("BITCORE_ENV_FILE"),
        ".env",
        os.path.join("workspace", ".env"),
    )

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            continue
        _apply_env_file(path)


def _apply_env_file(path: Path) -> None:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _ensure_dir(path: str) -> None:
    if not path:
        return
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _ensure_parent(path: str) -> None:
    if not path:
        return
    parent = Path(path).parent
    if parent == Path(path):
        return
    _ensure_dir(str(parent))


def _validate_required(config: Config) -> None:
    if config.DRY_RUN or os.getenv("PYTEST_CURRENT_TEST") or os.getenv("ALLOW_MISSING_SECRETS"):
        return

    missing: Dict[str, Optional[str]] = {
        "MORPHEUS_API_KEY": config.MORPHEUS_API_KEY,
        "CORE_API_KEY": config.CORE_API_KEY,
        "N8N_API_KEY": config.N8N_API_KEY,
    }

    unresolved = [name for name, value in missing.items() if not value]
    if unresolved:
        joined = ", ".join(unresolved)
        raise RuntimeError(
            f"Missing required environment variables: {joined}. "
            "Set DRY_RUN=true for local testing or populate the secrets."
        )


def get_config() -> Config:
    _load_env_files()

    morpheus_url = os.getenv("MORPHEUS_API_URL", "https://api.mor.org/api/v1/chat/completions")
    morpheus_key = os.getenv("MORPHEUS_API_KEY")

    core_url = os.getenv("CORE_API_URL", "https://api.core.ac.uk/")
    core_key = os.getenv("CORE_API_KEY")

    n8n_key = os.getenv("N8N_API_KEY")
    perception_webhook = os.getenv("PERCEPTION_WEBHOOK_URL", "https://n8n.bitwiki.org/webhook/agent-zero-final")

    log_file = os.getenv("LOG_FILE", "/a0/logs/research_coordinator.log")

    results_base = os.getenv("RESULTS_BASE_DIR", "/a0/bitcore/workspace/research_engine")
    results_raw = os.getenv("RESULTS_RAW_DIR", os.path.join(results_base, "raw_results"))

    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "/a0/workspace/research_results/knowledge_base.json")
    assurance_dir = os.getenv("ASSURANCE_REPORTS_DIR", "/a0/workspace/research_results/assurance_reports")

    tmp_dir = os.getenv("TMP_DIR", "/a0/tmp")
    dry_run = _env_bool("DRY_RUN", True)
    swarm_max_workers = _try_parse_int(os.getenv("SWARM_MAX_WORKERS"), 6)

    for path in (results_base, results_raw, assurance_dir, tmp_dir):
        _ensure_dir(path)
    for file_path in (log_file, kb_path):
        _ensure_parent(file_path)

    config = Config(
        MORPHEUS_API_URL=morpheus_url,
        MORPHEUS_API_KEY=morpheus_key,
        CORE_API_URL=core_url,
        CORE_API_KEY=core_key,
        N8N_API_KEY=n8n_key,
        PERCEPTION_WEBHOOK_URL=perception_webhook,
        LOG_FILE=log_file,
        RESULTS_BASE_DIR=results_base,
        RESULTS_RAW_DIR=results_raw,
        KNOWLEDGE_BASE_PATH=kb_path,
        ASSURANCE_REPORTS_DIR=assurance_dir,
        TMP_DIR=tmp_dir,
        SWARM_MAX_WORKERS=swarm_max_workers,
        DRY_RUN=dry_run,
    )

    _ensure_dir(config.ASSURANCE_REPORTS_DIR)
    _ensure_parent(config.KNOWLEDGE_BASE_PATH)
    _ensure_parent(config.LOG_FILE)
    _ensure_dir(config.RESULTS_RAW_DIR)
    _ensure_dir(config.TMP_DIR)

    _validate_required(config)
    return config


cfg = get_config()


def is_dry_run() -> bool:
    return cfg.DRY_RUN


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_bitcore_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handlers = []
    if cfg.LOG_FILE:
        try:
            file_handler = logging.FileHandler(cfg.LOG_FILE)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except OSError:
            pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    for handler in handlers:
        logger.addHandler(handler)

    logger.propagate = False
    setattr(logger, "_bitcore_configured", True)
    return logger


__all__ = ["cfg", "get_config", "get_logger", "is_dry_run", "Config"]
