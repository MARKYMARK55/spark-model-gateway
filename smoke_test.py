"""
smoke_test.py — Smoke test for the LiteLLM + vLLM + SparkRun local stack.

Tests four things in order:
  Step 1  vLLM health        — GET http://localhost:8000/v1/models
  Step 2  LiteLLM registry   — GET http://localhost:4000/v1/models (Bearer auth)
                               checks that the "local-coder" alias is registered
  Step 3  End-to-end chat    — POST through LiteLLM using the anthropic SDK
                               model="claude-sonnet-4-5", prints response + routed model
  Step 4  Thinking test      — Only for nemotron/qwen models (auto-detected).
                               Sends a reasoning prompt and checks for <think> in response.

How to run:
    python smoke_test.py
    python smoke_test.py --litellm-url http://localhost:4000 --vllm-url http://localhost:8000 --key simple-api-key

Exit codes:
    0  all mandatory steps passed (warns are OK)
    1  one or more steps failed
"""

import argparse
import sys

import requests
import anthropic


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Smoke test: LiteLLM + vLLM + SparkRun")
    p.add_argument("--litellm-url", default="http://localhost:4000",
                   help="LiteLLM base URL (default: http://localhost:4000)")
    p.add_argument("--vllm-url", default="http://localhost:8000",
                   help="vLLM base URL (default: http://localhost:8000)")
    p.add_argument("--key", default="simple-api-key",
                   help="LiteLLM API key (default: simple-api-key)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pass_(msg):
    print(f"  \u2713 PASS: {msg}")

def fail_(msg):
    print(f"  \u2717 FAIL: {msg}")

def warn_(msg):
    print(f"  \u26a0  WARN: {msg}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step1_vllm(vllm_url):
    """Check vLLM is serving a model."""
    print("[ STEP 1 ] Checking vLLM is serving a model ...")
    url = f"{vllm_url}/v1/models"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        if not models:
            fail_("vLLM responded but returned no models")
            return False, None
        model_id = models[0].get("id", "<unknown>")
        pass_(f"vLLM is serving model: {model_id}")
        return True, model_id
    except requests.exceptions.ConnectionError:
        fail_(f"Could not connect to vLLM at {url}")
        return False, None
    except Exception as e:
        fail_(f"Unexpected error: {e}")
        return False, None


def step2_litellm(litellm_url, api_key):
    """Check LiteLLM has local-coder registered."""
    print("[ STEP 2 ] Checking LiteLLM model registry for 'local-coder' alias ...")
    url = f"{litellm_url}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        count = len(models)
        ids = [m.get("id", "") for m in models]
        has_local_coder = any("local-coder" in mid for mid in ids)
        print(f"  Registered models ({count}): {ids}")
        if has_local_coder:
            pass_(f"'local-coder' alias found. Total models registered: {count}")
            return True
        else:
            fail_(f"'local-coder' alias NOT found in LiteLLM registry. Registered: {ids}")
            return False
    except requests.exceptions.ConnectionError:
        fail_(f"Could not connect to LiteLLM at {url}")
        return False
    except Exception as e:
        fail_(f"Unexpected error: {e}")
        return False


def step3_e2e(litellm_url, api_key):
    """End-to-end chat via anthropic SDK through LiteLLM."""
    print("[ STEP 3 ] Sending end-to-end chat request through LiteLLM (anthropic SDK) ...")
    client = anthropic.Anthropic(
        base_url=litellm_url,
        api_key=api_key,
    )
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": "Reply in one sentence: confirm you are running locally on DGX Spark.",
                }
            ],
        )
        text = msg.content[0].text if msg.content else ""
        routed_model = getattr(msg, "model", "<not reported>")
        print(f"  Response text : {text}")
        print(f"  Routed model  : {routed_model}")
        if text:
            pass_(f"Got a valid response. Routed to: {routed_model}")
            return True, routed_model
        else:
            fail_("Response was empty")
            return False, routed_model
    except Exception as e:
        fail_(f"anthropic SDK call failed: {e}")
        return False, None


def step4_thinking(litellm_url, api_key):
    """Optional thinking test for nemotron/qwen models."""
    print("[ STEP 4 ] Thinking test: sending reasoning prompt ...")
    client = anthropic.Anthropic(
        base_url=litellm_url,
        api_key=api_key,
    )
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": "What is 17 * 23? Show your reasoning step by step.",
                }
            ],
        )
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "thinking"):
                text += block.thinking
        has_think_tag = "<think>" in text.lower()
        if has_think_tag:
            pass_("<think> block detected in response — extended thinking is active")
            return True
        else:
            warn_("<think> block absent — extended thinking may be disabled or unsupported in this config")
            return True  # warn only, don't fail
    except Exception as e:
        fail_(f"Thinking test request failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    results = []

    # Step 1
    ok1, model_id = step1_vllm(args.vllm_url)
    results.append(ok1)

    # Step 2
    ok2 = step2_litellm(args.litellm_url, args.key)
    results.append(ok2)

    # Step 3
    ok3, routed_model = step3_e2e(args.litellm_url, args.key)
    results.append(ok3)

    # Step 4 — optional, auto-detected from vLLM model name
    step4_run = False
    if routed_model:
        name_lower = routed_model.lower()
        if "nemotron" in name_lower or "qwen" in name_lower:
            step4_run = True
    elif model_id:
        name_lower = model_id.lower()
        if "nemotron" in name_lower or "qwen" in name_lower:
            step4_run = True

    if step4_run:
        ok4 = step4_thinking(args.litellm_url, args.key)
        results.append(ok4)
    else:
        print("[ STEP 4 ] Thinking test: skipped (model is not nemotron/qwen)")

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    print()
    print(f"{'='*50}")
    print(f"  Summary: {passed}/{total} steps passed")
    print(f"{'='*50}")

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
