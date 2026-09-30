from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path


DEFAULTS = {
    "ARTICUTABLE_MLLM": (
        "YOUR_ARTICUTABLE_MLLM_MODEL", "YOUR_REASONING_EFFORT", "reasoning-only",
    ),
    "GRAM_MLLM": (
        "YOUR_GRAM_MLLM_MODEL", "YOUR_REASONING_EFFORT", "reasoning-only",
    ),
    "IMAGE_ORCHESTRATOR": (
        "YOUR_IMAGE_ORCHESTRATOR_MODEL", "YOUR_REASONING_EFFORT", "image-generation",
    ),
}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class ModelRuntime:
    domain: str
    model: str
    reasoning_effort: str
    execution_mode: str
    api_base: str
    api_key: str

    @property
    def provider(self):
        return "openai-api"

    def with_model(self, model):
        return replace(self, model=model or self.model)

    def with_execution_mode(self, execution_mode):
        mode = execution_mode or self.execution_mode
        if mode != self.execution_mode:
            raise ValueError(f"{self.domain} only supports {self.execution_mode}")
        return self

    def metadata(self):
        metadata = {
            "provider": "openai-api",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "execution_mode": self.execution_mode,
            "api_base": self.api_base,
            "api_key_env": "OPENAI_API_KEY",
        }
        if self.domain == "IMAGE_ORCHESTRATOR":
            metadata["image_generation_model"] = os.environ.get(
                "IMAGE_GENERATION_MODEL", "YOUR_IMAGE_GENERATION_MODEL",
            )
        return metadata


def load_runtime(domain, environment=None):
    environment = os.environ if environment is None else environment
    if domain not in DEFAULTS:
        raise ValueError(f"unknown model runtime domain: {domain}")
    default_model, default_effort, mode = DEFAULTS[domain]
    model = environment.get(f"{domain}_MODEL", default_model).strip()
    effort = environment.get(f"{domain}_REASONING_EFFORT", default_effort)
    api_base = environment.get("OPENAI_API_BASE", "YOUR_API_BASE").rstrip("/")
    api_key = environment.get("OPENAI_API_KEY", "YOUR_API_KEY").strip()
    if not model or model.startswith("YOUR_"):
        raise ValueError(f"{domain}_MODEL must be configured")
    if not effort or effort.startswith("YOUR_"):
        raise ValueError(f"{domain}_REASONING_EFFORT must be configured")
    if not api_base or api_base.startswith("YOUR_"):
        raise ValueError("OPENAI_API_BASE must be configured")
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported {domain} reasoning effort: {effort}")
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError("OPENAI_API_KEY is required")
    return ModelRuntime(domain, model, effort, mode, api_base, api_key)


def pipeline_runtime(environment=None):
    return load_runtime("ARTICUTABLE_MLLM", environment)


def gram_runtime(environment=None):
    return load_runtime("GRAM_MLLM", environment)


def image_generation_runtime(environment=None):
    return load_runtime("IMAGE_ORCHESTRATOR", environment)


def runtime_snapshot(environment=None):
    return {
        "articutable_mllm": pipeline_runtime(environment).metadata(),
        "gram_mllm": gram_runtime(environment).metadata(),
        "image_generation": image_generation_runtime(environment).metadata(),
    }


def write_runtime_marker(runtime, anchor):
    anchor = Path(anchor)
    marker = anchor.parent / f"{anchor.name}.runtime.json"
    marker.write_text(json.dumps(runtime.metadata(), indent=2) + "\n", encoding="utf-8")
    return marker


def image_data_url(path):
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def request_responses(runtime, payload, *, timeout=600, opener=urllib.request.urlopen):
    request = urllib.request.Request(
        runtime.api_base + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {runtime.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI Responses API failed ({exc.code}): {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI Responses API request failed: {exc.reason}") from exc


def _response_text(document):
    texts = [
        content["text"]
        for item in document.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text" and isinstance(content.get("text"), str)
    ]
    if not texts:
        raise RuntimeError("OpenAI Responses API returned no output text")
    return "\n".join(texts)


def run_mllm(
    runtime, prompt, images, response_path, cwd=None, *, stdout_path=None,
    stderr_path=None, opener=urllib.request.urlopen,
):
    if runtime.execution_mode != "reasoning-only":
        raise ValueError("run_mllm only supports reasoning-only requests")
    response_path = Path(response_path).resolve()
    response_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(stdout_path or response_path.with_suffix(".stdout.json"))
    stderr_path = Path(stderr_path or response_path.with_suffix(".stderr.txt"))
    content = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": image_data_url(path)} for path in images)
    payload = {
        "model": runtime.model,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": runtime.reasoning_effort},
        "tools": [],
        "tool_choice": "none",
        "store": False,
    }
    write_runtime_marker(runtime, response_path)
    document = request_responses(runtime, payload, opener=opener)
    stdout_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    text = _response_text(document)
    response_path.write_text(text, encoding="utf-8")
    return text
