"""Verify that the original project still matches the pre-copy SHA256 snapshot."""
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((root / 'docs/approved_source_snapshot.json').read_text(encoding='utf-8-sig'))
    original = Path(snapshot['original_root'])
    mismatches = []
    for item in snapshot['files']:
        path = original / item['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != item['sha256']:
            mismatches.append(item['path'])
    expected = {item['path'] for item in snapshot['files']}
    actual = {p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()
              and not any(part in {'.git','results','validation_logs','__pycache__'} or part.startswith('build')
                          for part in p.relative_to(original).parts)}
    report = dict(original_root=str(original), checked_files=len(expected),
                  changed_or_missing=mismatches, added=sorted(actual-expected),
                  original_unchanged=not mismatches and actual == expected)
    (root / 'docs/original_integrity_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['original_unchanged'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
