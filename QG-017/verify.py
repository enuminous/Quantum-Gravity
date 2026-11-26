from pathlib import Path
from hashlib import sha256
root=Path(__file__).resolve().parent
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert sha256((root/name).read_bytes()).hexdigest()==digest,name
    print('PASS',name)
