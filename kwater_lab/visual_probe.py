"""Compare two prompts on the organizer's published training example, using local inference.

Run from the project root with `python -m kwater_lab.visual_probe`.
Use `--prepare` on a new Windows x64 machine with a Vulkan GPU to fetch the assets.
The default cache contains the public example JPG, its published training label,
the two GGUFs from the pinned Qwen revision, and the extracted llama.cpp release.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from urllib.request import Request, urlopen
from zipfile import ZipFile


DEFAULT_CACHE = Path("G:/CodexData/kwater-rd-20261125/visual-probe")
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
MODEL_REV = "1cd86afb9a95c410a6038ab3b40d8b578c892266"
RUNTIME = "llama-b12721"
RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b12721"
OFFICIAL = "https://hackathon.imaginationgroup.co.kr/kwater/"
FILES = {
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf": "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
    "mmproj-Qwen3VL-4B-Instruct-F16.gguf": "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331",
    "public-example.jpg": "7a4e7cbe167c46753cf8b683f6a08a8983a43697ef59e97a00ad28c68e478d4d",
}
# Each condition sees the same alphabetically ordered codebook.
CODEBOOK = {
    "caught_in_machine": "기계 끼임·감김",
    "chemical_exposure": "유해물질 누출·접촉·흡입",
    "confined_space": "밀폐공간 진입",
    "drowning_inundation": "물에 빠짐·침수",
    "electrical_energy": "감전",
    "fall_height": "높이차 추락",
    "fire_explosion": "화재·폭발·압력 파열",
    "other": "기타 관측 위험",
    "slip_trip": "같은 높이 미끄러짐·걸림",
    "struck_by_lifting": "낙하·비래·인양물 맞음",
    "thermal_weather": "고온·저온·기상 노출",
}
INSTRUCTION = (
    "현장 사진 한 장을 분석하세요. 사진 속 위험을 다음 목록의 서로 다른 코드 0~3개로 고르고 "
    "선택한 각 코드의 근거를 한국어 한 문장으로 쓰세요. 관측된 위험이 없으면 빈 목록을 반환하세요.\n"
    + "\n".join(f"{code} = {name}" for code, name in CODEBOOK.items())
)
CONDITIONS = {
    "direct": INSTRUCTION,
    "evidence_gate": INSTRUCTION
    + "\n각 후보에 대해 사진 속 위치, 대상 및 위험한 상태를 모두 특정할 수 있는지 먼저 확인하고, "
    "그 근거가 부족하거나 보이지 않는 조건을 추측해야 하는 후보는 제외하세요.",
}
SCHEMA = {
    "type": "object",
    "properties": {
        "hazards": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "enum": list(CODEBOOK)},
                    "evidence": {"type": "string"},
                },
                "required": ["code", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hazards"],
    "additionalProperties": False,
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def fetch_file(url, path, expected):
    if path.is_file() and digest(path) == expected:
        return
    partial = path.with_suffix(path.suffix + ".part")
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as response:
        with partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    if digest(partial) != expected:
        raise ValueError(f"Unexpected SHA-256 from {url}")
    partial.replace(path)


def prepare(cache):
    cache.mkdir(parents=True, exist_ok=True)
    model_base = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/"
    for name, expected in FILES.items():
        url = OFFICIAL + "kwater-data-example.jpg" if name == "public-example.jpg" else model_base + name
        fetch_file(url, cache / name, expected)
    archive = cache / f"{RUNTIME}-bin-win-vulkan-x64.zip"
    fetch_file(f"{RELEASE}/{archive.name}", archive,
               "2e15aaf92807ef03b64c38ac52ad403770d4a67079ddfd00808bd86c51e20f4e")
    with ZipFile(archive) as zipped:
        zipped.extractall(cache / RUNTIME)
    guide = Request(OFFICIAL + "api/report", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(guide, timeout=30) as stream:
        sections = json.load(stream)["participant"]["sections"]
    for section in sections:
        for item in section.get("briefing", []):
            for block in item.get("doc", []):
                code = block.get("code", "")
                if '"sample_id": "S16C159"' in code:
                    label = json.loads(code)
                    (cache / "public-example-label.json").write_text(
                        json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
                    return
    raise ValueError("The organizer's guide no longer provides the pinned sample label")


def local_request(port, api_key, path, body=None, timeout=120):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        connection.request("POST" if data else "GET", path, body=data, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class GenerationRequestError(RuntimeError):
    def __init__(self, status, response, elapsed):
        self.status = status
        detail = response.get("error") if isinstance(response, dict) else None
        self.details = detail if isinstance(detail, dict) else {}
        self.elapsed = elapsed
        super().__init__(f"Local inference failed with HTTP {status}: {self.details.get('type', 'unknown')}")


def generate(port, api_key, data_image, instruction, schema, max_tokens):
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_image}},
            {"type": "text", "text": instruction},
        ]}],
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "observed_hazards", "strict": True, "schema": schema,
        }},
    }
    start = time.perf_counter()
    status, response = local_request(port, api_key, "/v1/chat/completions", payload, timeout=300)
    elapsed = time.perf_counter() - start
    if status != 200:
        raise GenerationRequestError(status, response, elapsed)
    choice = response["choices"][0]
    return {
        "prompt": instruction, "raw_output": choice["message"]["content"],
        "finish_reason": choice["finish_reason"], "elapsed_s": round(elapsed, 3),
        "usage": response.get("usage"), "timings": response.get("timings"),
    }


def run_trial(port, api_key, data_image, instruction):
    result = generate(port, api_key, data_image, instruction, SCHEMA, 384)
    try:
        hazards = json.loads(result["raw_output"])["hazards"]
        predicted = [item["code"] for item in hazards]
        valid = (result["finish_reason"] == "stop" and len(predicted) <= 3
                 and len(set(predicted)) == len(predicted)
                 and all(item["code"] in CODEBOOK and bool(item["evidence"].strip()) for item in hazards))
    except (KeyError, TypeError, AttributeError, ValueError):
        predicted, valid = None, False
    result.update(predicted_codes=predicted, valid=valid)
    return result


def encoded_image(image):
    return "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")


@contextmanager
def local_server(cache, log, *, weights=None, runtime_root=None, extra_flags=(), startup_timeout=300, context_size=4096):
    weights = weights or {name: sha for name, sha in FILES.items() if name.endswith(".gguf")}
    model, projector = [cache / name for name in weights]
    for name, expected in weights.items():
        if digest(cache / name) != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
    server = (runtime_root or cache) / RUNTIME / ("llama-server.exe" if os.name == "nt" else "llama-server")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    flags = ["--offline", "-ngl", "99", "-c", str(context_size), "-np", "1", "--no-webui", "--host", "127.0.0.1", *extra_flags]
    command = [str(server), *flags, "--port", str(port), "-m", str(model), "--mmproj", str(projector)]
    env = os.environ.copy()
    env["LLAMA_API_KEY"] = key
    start = time.perf_counter()
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                status = process.poll()
                if status is not None:
                    raise RuntimeError(f"Server exited during startup with status {status}. Inspect {log}")
                if time.perf_counter() - start > startup_timeout:
                    raise TimeoutError(f"Server was not ready within {startup_timeout} seconds. Inspect {log}")
                try:
                    if local_request(port, key, "/health", timeout=2)[0] == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.25)
            startup = round(time.perf_counter() - start, 3)
            yield port, key, {"startup_s": startup, "server_flags": flags, "server_sha256": digest(server)}
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prepare", action="store_true", help="fetch pinned public Windows x64 assets before inference")
    parser.add_argument("--output", type=Path, default=Path("results/visual_probe.json"))
    args = parser.parse_args()
    cache = args.cache.resolve()
    if args.prepare:
        prepare(cache)
    image = cache / "public-example.jpg"
    if digest(image) != FILES[image.name]:
        raise ValueError("Public sample image checksum changed")
    data_image = encoded_image(image)
    with local_server(cache, cache / "visual-probe-server.log") as (port, key, startup):
        trials = {name: run_trial(port, key, data_image, prompt) for name, prompt in CONDITIONS.items()}
    label = json.loads((cache / "public-example-label.json").read_text(encoding="utf-8"))
    expected = set(label["target"]["hazards"])
    for result in trials.values():
        if result["valid"]:
            predicted = set(result["predicted_codes"])
            result["matched"] = sorted(expected & predicted)
            result["missed"] = sorted(expected - predicted)
            result["extra"] = sorted(predicted - expected)
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "One public training illustration, two zero-shot prompts, same image and weights",
        "model": MODEL_REPO, "revision": MODEL_REV, "runtime": RUNTIME, "backend": "Vulkan",
        "official_competition_lora": False, "trained_on_competition_data_in_this_run": False,
        "model_url": f"https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REV}",
        "runtime_url": "https://github.com/ggml-org/llama.cpp/releases/tag/b12721",
        "image_url": "https://hackathon.imaginationgroup.co.kr/kwater/kwater-data-example.jpg",
        "label_url": "https://hackathon.imaginationgroup.co.kr/kwater/#core-data",
        "sha256": FILES, **startup,
        "label_sha256": digest(cache / "public-example-label.json"), "sample_id": label["sample_id"],
        "expected_codes": sorted(expected), "host": platform.platform(),
        "seed": 42, "temperature": 0, "max_tokens": 384,
        "conditions": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for name, result in trials.items():
        print(name, result["predicted_codes"], f'{result["elapsed_s"]} s', "valid" if result["valid"] else "invalid")


if __name__ == "__main__":
    main()
