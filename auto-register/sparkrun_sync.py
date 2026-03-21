#!/usr/bin/env python3
"""
sparkrun_sync.py
================
Auto-detect running SparkRun / vLLM models and register/deregister LiteLLM
presets and claude-* aliases dynamically via the LiteLLM database API.

No proxy restart needed — models appear and disappear in real time as SparkRun
loads and unloads them.

Usage
-----
  python sparkrun_sync.py --once               # single sync pass then exit
  python sparkrun_sync.py --watch              # continuous polling (default 30s)
  python sparkrun_sync.py --watch --interval 15
  python sparkrun_sync.py --deregister-all     # remove all dynamically registered models
  python sparkrun_sync.py --once --port 8001   # sync only one endpoint

Requirements: pip install requests pyyaml

Configuration
-------------
  Edit models.yaml to define which models map to which presets and claude aliases.
  Edit endpoints: in models.yaml to set which ports to poll.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (override via CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

LITELLM_URL = "http://localhost:4000"
LITELLM_KEY = "simple-api-key"
MODELS_YAML  = Path(__file__).parent / "models.yaml"
STATE_FILE   = Path("/tmp/sparkrun_sync_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sparkrun_sync")


# ─────────────────────────────────────────────────────────────────────────────
# Config & state helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"ports": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# vLLM endpoint detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_model(port: int) -> str | None:
    """Query a vLLM endpoint and return the served model name, or None if down."""
    try:
        resp = requests.get(
            f"http://localhost:{port}/v1/models",
            headers={"Authorization": "Bearer EMPTY"},
            timeout=5,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return models[0]["id"] if models else None
    except Exception:
        return None


def find_model_config(model_name: str, config: dict) -> dict | None:
    """Look up a model in models.yaml. Exact match first, then substring."""
    models = config.get("models", {})
    if model_name in models:
        return models[model_name]
    for pattern, cfg in models.items():
        if pattern in model_name or model_name in pattern:
            return cfg
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Thinking style translation
# ─────────────────────────────────────────────────────────────────────────────

def build_thinking_params(
    thinking_value: int | None,
    thinking_style: str,
    preset_name: str,
    model_cfg: dict,
) -> dict:
    """Translate a thinking budget into model-family-specific litellm_params."""
    if not thinking_value or thinking_style == "none":
        return {}

    if thinking_style == "thinking_budget":
        # Qwen3 / Qwen3.5 / MiniMax — top-level param
        return {"thinking_budget": thinking_value}

    elif thinking_style == "nemotron":
        # NVIDIA Nemotron — passed via extra_body
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "thinking_budget": thinking_value,
                }
            }
        }

    elif thinking_style == "effort":
        # GPT-OSS and similar — budget values map to effort levels
        effort_map = model_cfg.get("effort_map", {
            "Expert": "medium", "Heavy": "high", "Max": "max", "Code": "medium", "Creative": "low",
        })
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "effort": effort_map.get(preset_name, "medium"),
                }
            }
        }

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# LiteLLM API
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {LITELLM_KEY}",
        "Content-Type": "application/json",
    }


def register_model(name: str, litellm_params: dict, description: str = "") -> str | None:
    """POST /model/new — returns model_id or None on failure."""
    payload: dict[str, Any] = {"model_name": name, "litellm_params": litellm_params}
    if description:
        payload["model_info"] = {"description": description}
    try:
        resp = requests.post(f"{LITELLM_URL}/model/new", headers=_headers(), json=payload, timeout=10)
        resp.raise_for_status()
        model_id = resp.json().get("model_id", "")
        log.info("  + %-35s id: %s", name, str(model_id)[:12])
        return model_id or "unknown"
    except requests.RequestException as e:
        log.error("  ✗ Failed to register %s: %s", name, e)
        return None


def deregister_model(model_id: str, name: str) -> None:
    """POST /model/delete — removes a model from the LiteLLM DB."""
    try:
        requests.post(
            f"{LITELLM_URL}/model/delete",
            headers=_headers(),
            json={"id": model_id},
            timeout=10,
        ).raise_for_status()
        log.info("  - Removed: %s", name)
    except requests.RequestException as e:
        log.error("  ✗ Failed to remove %s: %s", name, e)


# ─────────────────────────────────────────────────────────────────────────────
# Preset registration
# ─────────────────────────────────────────────────────────────────────────────

def register_presets(
    vllm_model_name: str,
    model_cfg: dict,
    port: int,
    config: dict,
    endpoint_cfg: dict,
) -> dict[str, str]:
    """Register all presets (and optional claude alias) for a detected model."""
    defaults       = config.get("defaults", {})
    all_presets    = config.get("presets", {})
    short_name     = model_cfg.get("short_name", vllm_model_name.split("/")[-1])
    thinking_style = model_cfg.get("thinking_style", defaults.get("thinking_style", "thinking_budget"))
    overrides      = model_cfg.get("override", {})
    api_base       = model_cfg.get("api_base", defaults.get("api_base", f"http://host.docker.internal:{port}/v1"))
    api_key        = model_cfg.get("api_key",  defaults.get("api_key", "EMPTY"))

    # Which presets to register
    allowed  = endpoint_cfg.get("presets", "all")
    per_model = model_cfg.get("presets")
    if per_model:
        preset_names = per_model
    elif allowed == "all" or allowed is None:
        preset_names = list(all_presets.keys())
    else:
        preset_names = allowed

    registered: dict[str, str] = {}

    for preset_name in preset_names:
        if preset_name not in all_presets:
            continue

        preset = all_presets[preset_name]
        display_name = f"{short_name}-{preset_name}"

        params: dict[str, Any] = {
            "model":    f"openai/{vllm_model_name}",
            "api_base": api_base,
            "api_key":  api_key,
        }
        for key in ("temperature", "top_p", "max_tokens", "timeout", "stream_timeout"):
            val = overrides.get(key, preset.get(key))
            if val is not None:
                params[key] = val

        params.update(build_thinking_params(preset.get("thinking"), thinking_style, preset_name, model_cfg))

        model_id = register_model(display_name, params, f"{short_name} — {preset_name} preset")
        if model_id:
            registered[display_name] = model_id

    # Optional claude-* alias
    claude_alias = model_cfg.get("claude_code_alias")
    if claude_alias:
        expert = all_presets.get("Expert", all_presets.get("Fast", {}))
        params = {
            "model":    f"openai/{vllm_model_name}",
            "api_base": api_base,
            "api_key":  api_key,
            "temperature":  overrides.get("temperature", expert.get("temperature", 0.55)),
            "top_p":        expert.get("top_p", 0.98),
            "max_tokens":   overrides.get("max_tokens", expert.get("max_tokens", 32768)),
            "timeout":      400,
            "stream_timeout": 400,
        }
        params.update(build_thinking_params(expert.get("thinking", 4096), thinking_style, "Expert", model_cfg))

        for alias in [claude_alias, "local-coder"]:
            mid = register_model(alias, params, f"Claude alias → {short_name} (local)")
            if mid:
                registered[alias] = mid

    return registered


def register_utility(vllm_model_name: str, util_cfg: dict, port: int) -> dict[str, str]:
    """Register a utility model (judge, router, embedder) — no presets."""
    register_as = util_cfg.get("register_as", vllm_model_name)
    params = {
        "model":    f"openai/{vllm_model_name}",
        "api_base": f"http://host.docker.internal:{port}/v1",
        "api_key":  "EMPTY",
        "temperature": 0.1,
        "max_tokens":  4096,
        "timeout":     60,
        "stream_timeout": 60,
    }
    registered: dict[str, str] = {}
    for name in [register_as] + util_cfg.get("aliases", []):
        mid = register_model(name, params, util_cfg.get("description", ""))
        if mid:
            registered[name] = mid
    return registered


# ─────────────────────────────────────────────────────────────────────────────
# Deregistration helpers
# ─────────────────────────────────────────────────────────────────────────────

def deregister_port(port: int, state: dict) -> None:
    port_key  = str(port)
    models    = state.get("ports", {}).get(port_key, {}).get("models", {})
    if not models:
        return
    log.info("Removing %d models from port %d...", len(models), port)
    for name, mid in models.items():
        deregister_model(mid, name)
    state.setdefault("ports", {})[port_key] = {"vllm_model": None, "models": {}}
    save_state(state)


def deregister_all(state: dict) -> None:
    for port_key in list(state.get("ports", {}).keys()):
        deregister_port(int(port_key), state)
    state["ports"] = {}
    save_state(state)
    log.info("All dynamic models deregistered.")


# ─────────────────────────────────────────────────────────────────────────────
# Main sync loop
# ─────────────────────────────────────────────────────────────────────────────

def sync_once(config: dict, state: dict, force_port: int | None = None) -> None:
    """One sync pass. If force_port is set, only check that endpoint."""
    endpoints      = config.get("endpoints", {})
    utility_models = config.get("utility_models", {})

    for ep_name, ep_cfg in endpoints.items():
        port = ep_cfg["port"]
        if force_port and port != force_port:
            continue

        role     = ep_cfg.get("role", "primary")
        port_key = str(port)
        detected = detect_model(port)
        prev     = state.get("ports", {}).get(port_key, {}).get("vllm_model")

        if detected == prev:
            continue

        if detected is None:
            log.info("Port %d: %s went offline", port, prev)
            deregister_port(port, state)
            continue

        if prev:
            log.info("Port %d: model changed  %s → %s", port, prev, detected)
            deregister_port(port, state)
        else:
            log.info("Port %d: detected %s", port, detected)

        # Utility model?
        registered: dict[str, str] = {}
        is_utility = False
        if role == "utility":
            for _, util_cfg in utility_models.items():
                patterns = util_cfg.get("model_patterns", [])
                if any(p.lower() in detected.lower() for p in patterns):
                    registered = register_utility(detected, util_cfg, port)
                    is_utility = True
                    break

        if not is_utility:
            model_cfg = find_model_config(detected, config)
            if model_cfg:
                log.info("  Config match: %s", model_cfg.get("short_name", "?"))
                registered = register_presets(detected, model_cfg, port, config, ep_cfg)
            else:
                # Unknown model — register with a generic name, no presets
                log.warning("  No config for '%s' — registering as generic", detected)
                generic = detected.split("/")[-1]
                params = {
                    "model":    f"openai/{detected}",
                    "api_base": f"http://host.docker.internal:{port}/v1",
                    "api_key":  "EMPTY",
                    "max_tokens": 32768,
                    "timeout":    300,
                }
                mid = register_model(generic, params, f"Auto-detected on port {port}")
                if mid:
                    registered[generic] = mid

        state.setdefault("ports", {})[port_key] = {"vllm_model": detected, "models": registered}
        save_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global LITELLM_URL, LITELLM_KEY  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="SparkRun → LiteLLM auto-registration sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--once",           action="store_true", help="Single sync pass then exit")
    parser.add_argument("--watch",          action="store_true", help="Continuous polling mode")
    parser.add_argument("--interval",       type=int, default=30, metavar="SEC", help="Poll interval (default: 30s)")
    parser.add_argument("--deregister-all", action="store_true", help="Remove all dynamic models and exit")
    parser.add_argument("--port",           type=int, help="Limit sync to a single endpoint port")
    parser.add_argument("--litellm-url",    default=LITELLM_URL, help="LiteLLM base URL")
    parser.add_argument("--litellm-key",    default=LITELLM_KEY, help="LiteLLM master key")
    parser.add_argument("-v", "--verbose",  action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    LITELLM_URL = args.litellm_url
    LITELLM_KEY = args.litellm_key

    config = load_config()
    state  = load_state()

    if args.deregister_all:
        deregister_all(state)
        return

    if args.once:
        sync_once(config, state, force_port=args.port)
        return

    # Default: watch mode
    log.info("Polling every %ds — Ctrl+C to stop", args.interval)
    try:
        while True:
            sync_once(config, state, force_port=args.port)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
