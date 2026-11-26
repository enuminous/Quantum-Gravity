from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    h,n=line.split('  ',1); assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    print('OK',n)
