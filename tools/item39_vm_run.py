"""Drive the local SSH guest and timestamp benchmark events on the Windows host."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("counts", type=int, nargs="*", default=[3, 200, 300])
a = ap.parse_args()
if not set(a.counts) <= {3, 200, 300}:
    raise ValueError("Expected a requested benchmark cohort")
vm = a.run / "vm"
identity = [
    "-i",
    str(vm / "guest-login"),
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=75",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    f"UserKnownHostsFile={vm / 'known_hosts'}",
]
remote = "runner@127.0.0.1"
ssh = ["ssh", "-p", "22439", *identity, remote]
probe = "uname -a; cat /etc/os-release; lscpu; free -h; df -h /; date -u"
attempt = 0
while True:
    attempt += 1
    result = subprocess.run([*ssh, probe], capture_output=True)
    if result.returncode == 0:
        (a.run / "linux-host.log").write_bytes(result.stdout)
        break
    message = result.stderr.decode(errors="replace")
    if "Permission denied" in message or "HOST IDENTIFICATION HAS CHANGED" in message:
        raise RuntimeError(message)
    print(
        "Waiting for guest",
        attempt,
        message[-140:],
        flush=True,
    )
    time.sleep(30)
events = {}
benchmarks = a.run / "linux"
benchmarks.mkdir(exist_ok=True)


def collect(count):
    dest = benchmarks / f"n{count}"
    dest.mkdir(exist_ok=True)
    cmd = [
        "scp",
        "-p",
        "-P",
        "22439",
        *identity,
        remote + f":/home/runner/item39/n{count}/*",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    guest = json.loads((dest / "timing.json").read_bytes())
    e = events[count]
    host = dict(
        install_s=e["install_end"] - e["install_start"],
        main_s=e["main_exit"] - e["main_start"],
        wall_install_start_to_exit_s=e["main_exit"] - e["install_start"],
    )
    host["guest_install_plus_process_s"] = guest["install_plus_process_s"]
    host["receipt_clock"] = (
        "Windows time.perf_counter; guest flushed markers over loopback SSH"
    )
    (dest / "host-timing.json").write_text(json.dumps(host, indent=2), encoding="utf8")
    print("COLLECTED", count, host, flush=True)


command = (
    "curl -fsSL http://10.0.2.2:18439/prepare.sh -o /tmp/kwater39-prepare.sh && sudo bash /tmp/kwater39-prepare.sh "
    + " ".join(map(str, a.counts))
)
with (
    (a.run / "linux-session.log").open("a", encoding="utf8", newline="\n") as log,
    concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
):
    jobs = []
    with subprocess.Popen(
        [*ssh, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf8",
        errors="replace",
    ) as process:
        for line in process.stdout:
            stamp = time.perf_counter()
            log.write(line)
            log.flush()
            fields = line.split()
            if len(fields) == 3 and fields[0] == "BENCH_EVENT":
                events.setdefault(int(fields[1]), {})[fields[2]] = stamp
                print(line.strip(), flush=True)
            if len(fields) == 2 and fields[0] == "ARTIFACTS_READY":
                jobs.append(pool.submit(collect, int(fields[1])))
        status = process.wait()
    for job in jobs:
        job.result()
    if status:
        raise RuntimeError(f"Guest benchmark exited {status}; see linux-session.log")
