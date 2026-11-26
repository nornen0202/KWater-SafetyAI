import io
import importlib
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "hostmods"))
pycdlib = importlib.import_module("pycdlib")

key = root / "guest-login"
if not key.exists():
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True
    )
config = """#cloud-config
users:
  - name: runner
    groups: [sudo]
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - KEY
ssh_pwauth: false
disable_root: true
growpart:
  mode: auto
  devices: ['/']
resize_rootfs: true
""".replace("KEY", key.with_suffix(".pub").read_text().strip())
iso = pycdlib.PyCdlib()
iso.new(interchange_level=3, joliet=3, rock_ridge="1.09", vol_ident="cidata")
for i, (name, s) in enumerate(
    [
        ("user-data", config),
        ("meta-data", "instance-id: item39-v1\nlocal-hostname: kwater39\n"),
    ]
):
    b = s.encode()
    iso.add_fp(
        io.BytesIO(b),
        len(b),
        iso_path=f"/F{i}.;1",
        rr_name=name,
        joliet_path="/" + name,
    )
iso.write(str(root / "seed.iso"))
iso.close()
