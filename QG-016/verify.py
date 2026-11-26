from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=digest:raise SystemExit('FAIL '+name)
    print(name+': OK')
