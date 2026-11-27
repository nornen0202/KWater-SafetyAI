"""Offline two-slot llama.cpp timing entrypoint for the pinned 27B vision GGUF."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import http.client
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time

from PIL import Image, ImageOps
from format_runtime import ActionSupportedLawRetriever, format_v3_record, parse_json_loose, validate_vlm_11code_obj


class LocalConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=240)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def call(path, method, endpoint, payload=None):
    conn = LocalConnection(path)
    try:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        conn.request(method, endpoint, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def classify(row, *, files, sock, prompt):
    image_id = row["image_id"]
    with Image.open(files[image_id]) as source:
        pic = ImageOps.exif_transpose(source).convert("RGB")
        scale = min(1.0, math.sqrt(393216 / (pic.width * pic.height)))
        pic.thumbnail((max(1, int(pic.width * scale)), max(1, int(pic.height * scale))))
        image = io.BytesIO()
        pic.save(image, "JPEG", quality=92)
    image_data = base64.b64encode(image.getvalue()).decode("ascii")
    payload = {
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_data}},
            {"type": "text", "text": "이 현장사진을 평가하고 지정된 JSON 하나만 출력한다."}]}],
        "temperature": 0, "max_tokens": 620, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, reply = call(sock, "POST", "/v1/chat/completions", payload)
    if status != 200:
        raise RuntimeError(f"vision generation failed with HTTP {status}: {str(reply)[:160]}")
    parsed, err = parse_json_loose(reply["choices"][0]["message"]["content"])
    return image_id, parsed, err, reply["usage"], reply["choices"][0].get("finish_reason")


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text("utf-8").splitlines() if s.strip()]


def main(input_dir, output_dir):
    root = Path(__file__).resolve().parent
    rows = read_jsonl(input_dir / "manifest.jsonl")
    images = {p.stem: p for p in (input_dir / "images").iterdir() if p.is_file()}
    prompt = (root / "system_prompt_v3.txt").read_text("utf-8")
    law = ActionSupportedLawRetriever(read_jsonl(root / "assets/laws.jsonl"),
                                      read_jsonl(root / "assets/cases_11code.jsonl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        sock = Path(tmp) / "vision.sock"
        binary = Path(tmp) / "llama-server"
        shutil.copyfile(root / "runtime/llama-server", binary)
        binary.chmod(0o700)
        for library in (root / "runtime").glob("*.so*"):
            (binary.parent / library.name).symlink_to(library)
        lib = str(root / "runtime")
        env = {**os.environ, "LD_LIBRARY_PATH": lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        command = [str(binary), "--host", str(sock), "--model", str(root / "model.gguf"),
                   "--mmproj", str(root / "mmproj.gguf"), "-ngl", "99", "--parallel", "2",
                   "--ctx-size", "8192", "--threads", "4", "--cache-ram", "0", "--reasoning-budget", "0"]
        server = subprocess.Popen(command, env=env)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load: {server.returncode}")
                try:
                    if call(sock, "GET", "/health")[0] == 200:
                        break
                except (FileNotFoundError, ConnectionRefusedError):
                    pass
                time.sleep(0.3)
            else:
                raise TimeoutError("vision model load exceeded four minutes")
            print(f"GGUF_READY seconds={time.monotonic()-started:.2f}", flush=True)
            job = partial(classify, files=images, sock=sock, prompt=prompt)
            parsed_count = schema_valid = tokens = prompt_tokens = capped = 0
            with ThreadPoolExecutor(max_workers=2) as workers, (output_dir / "submission.jsonl").open("w", encoding="utf-8") as out:
                for count, (iid, obj, err, usage, reason) in enumerate(workers.map(job, rows), 1):
                    parsed_count += int(err is None)
                    schema_valid += int(err is None and not validate_vlm_11code_obj(obj))
                    tokens += usage["completion_tokens"]
                    prompt_tokens += usage["prompt_tokens"]
                    capped += int(reason == "length")
                    out.write(json.dumps(format_v3_record(iid, obj, law), ensure_ascii=False) + "\n")
                    out.flush()
                    if count % 10 == 0 or count == len(rows):
                        print(f"GGUF_ROWS {count}/{len(rows)} parsed={parsed_count} schema_valid={schema_valid} "
                              f"tokens={tokens} prompt_tokens={prompt_tokens} capped={capped} "
                              f"seconds={time.monotonic()-started:.2f}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    cli = parser.parse_args()
    main(cli.input_dir, cli.output_dir)
