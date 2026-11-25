"""Print the paginated Markdown report to a Korean-font PDF using a local Chromium browser."""

import argparse
import base64
import html
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote, unquote

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/nornen0202/KWater-SafetyAI/blob/main/"
CSS = """
@page { size:A4; margin:0; }
* { box-sizing:border-box; }
body { margin:0; color:#193244; font:11.7pt/1.64 "Malgun Gothic","Noto Sans CJK KR",sans-serif;
       -webkit-print-color-adjust:exact; print-color-adjust:exact; }
.paper { position:relative; width:210mm; min-height:296.75mm; padding:14mm 16mm 18mm;
         break-after:page; }
.paper:last-child { break-after:auto; }
.running { color:#4c6a73; display:flex; justify-content:space-between; padding-bottom:2.8mm;
           border-bottom:1.5pt solid #007c76; font-size:8.6pt; font-weight:600; }
footer { position:absolute; left:16mm; right:16mm; bottom:8mm; display:flex; justify-content:space-between;
         border-top:.6pt solid #c6d5d7; padding-top:2mm; font-size:8pt; color:#496776; }
h1,h2,h3 { color:#16384a; break-after:avoid; word-break:keep-all; }
h1 { font-size:28pt; line-height:1.25; margin:8mm 0 4mm; letter-spacing:-1.5pt; }
h2 { font-size:22pt; line-height:1.35; margin:5mm 0 3.5mm; letter-spacing:-.8pt; }
h3 { font-size:13.2pt; margin:4mm 0 1.4mm; color:#006c67; }
p { margin:1.8mm 0 2.8mm; }
p,td,th,li { word-break:keep-all; overflow-wrap:anywhere; }
strong { font-weight:700; }
blockquote { border-left:3pt solid #00857a; background:#eaf5f2; padding:2.5mm 4.5mm; margin:4mm 0; }
blockquote p { margin:0; font-size:12pt; line-height:1.6; }
table { width:100%; border-collapse:collapse; font-size:10.3pt; line-height:1.48; margin:3mm 0; }
th,td { text-align:left; vertical-align:top; padding:1.8mm 2mm; border-bottom:.5pt solid #cedbdc; }
th { background:#e6efee; color:#16453f; }
tr { break-inside:avoid; }
img { max-width:100%; max-height:93mm; display:block; margin:2mm auto 0; object-fit:contain; }
p:has(>img:only-child) { margin:2mm 0 1mm; }
p:has(>em:first-child) { font-size:9.35pt; color:#526974; margin:0 0 3mm; }
em { font-style:normal; }
a { color:#076963; text-decoration:underline; text-underline-offset:2px; }
ul,ol { padding-left:5.4mm; margin:1.8mm 0 3mm; }
li { margin:.8mm 0; }
code { font-family:"Consolas","Malgun Gothic",monospace; font-size:.93em; }
"""


def print_pdf(browser, source, dest, tmp):
    with tempfile.NamedTemporaryFile(dir=tmp, prefix="research-", suffix=".pdf", delete=False) as out:
        target = Path(out.name)
    command = [str(browser), "--headless", "--disable-gpu", "--disable-extensions", "--no-first-run",
               "--disable-background-networking", "--no-pdf-header-footer", f"--user-data-dir={tmp/'profile'}",
               f"--print-to-pdf={target}", source.as_uri()]
    try:
        with subprocess.Popen(command) as process:
            deadline = time.monotonic() + 180
            try:
                while time.monotonic() < deadline:
                    if target.stat().st_size > 32:
                        with target.open("rb") as raw:
                            header = raw.read(5)
                            raw.seek(-32, 2)
                            complete = header == b"%PDF-" and raw.read().rstrip().endswith(b"%%EOF")
                        if complete:
                            try:
                                target.replace(dest)
                                return
                            except PermissionError:
                                if process.poll() is not None:
                                    raise
                    if process.poll() is not None:
                        raise RuntimeError(f"Browser exited before printing a complete PDF ({process.returncode})")
                    time.sleep(.5)
                raise TimeoutError("Browser did not finish writing a PDF in 180 seconds")
            finally:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
    finally:
        target.unlink(missing_ok=True)


def render(source, dest, browser, tmp):
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    sections = source.read_text(encoding="utf-8").split("<!-- page -->")
    pages = []
    for index, part in enumerate(sections, 1):
        tokens = md.parse(part)
        for token in tokens:
            for t in token.children or []:
                if t.type == "image":
                    p = (source.parent / unquote(t.attrGet("src"))).resolve()
                    if p.suffix != ".png" or not p.is_relative_to(source.parent.resolve()):
                        raise ValueError(f"Unexpected report image: {p}")
                    t.attrSet("src","data:image/png;base64," + base64.b64encode(p.read_bytes()).decode())
                if t.type == "link_open":
                    href = t.attrGet("href")
                    if not href.startswith(("https://","http://","#")):
                        p = (source.parent / unquote(href)).resolve()
                        if not p.is_file():
                            raise ValueError(f"Missing report source: {p}")
                        t.attrSet("href",GITHUB + quote(p.relative_to(ROOT).as_posix()))
        body = md.renderer.render(tokens,md.options,{})
        pages.append('<article class="paper"><div class="running"><span>K-water 현장 안전 AI</span>'
                     '<span>연구·평가 보고서</span></div>' + body +
                     f'<footer><span>2026년 11월 25일 기준 · 연구팀 실험 기록</span><span>{index} / {len(sections)}</span></footer></article>')
    tmp.mkdir(parents=True,exist_ok=True)
    print_source = tmp / "research-print.html"
    print_source.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>' +
                            html.escape("K-water 현장 안전 AI 연구·평가") + "</title><style>" + CSS +
                            "</style><body>" + "".join(pages) + "</body></html>",encoding="utf-8")
    dest.parent.mkdir(parents=True,exist_ok=True)
    print_pdf(browser, print_source.resolve(), dest, tmp)
    print(f"Printed {len(sections)} planned pages to {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"reports/competition-research-2026-11-25.md")
    p.add_argument("--output",type=Path,default=ROOT/"reports/competition-research-2026-11-25.pdf")
    p.add_argument("--browser",type=Path,default=Path(shutil.which("chromium") or
                   "C:/Program Files/Google/Chrome/Application/chrome.exe"))
    args=p.parse_args()
    render(args.source.resolve(),args.output,args.browser,ROOT/".runtime/report-qa")
