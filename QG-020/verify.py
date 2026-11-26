from pathlib import Path
import hashlib
p=Path(__file__).resolve().parent
for line in (p/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1)
    assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
    print('OK',n)
