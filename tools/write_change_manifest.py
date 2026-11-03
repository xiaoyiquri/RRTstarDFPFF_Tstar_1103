"""Inventory modifications against the approved baseline snapshot."""
import hashlib
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    snapshot=json.loads((root/"docs/approved_source_snapshot.json").read_text(encoding="utf-8-sig"))
    baseline={f["path"]:f["sha256"] for f in snapshot["files"]}
    paths={p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file() and
           not any(part.startswith("build") or part in {".git","results","validation_logs","__pycache__",
               ".python_packages",".venv"} for part in p.relative_to(root).parts)}
    changed=[n for n,h in baseline.items() if n in paths and hashlib.sha256(paths[n].read_bytes()).hexdigest()!=h]
    report=dict(baseline=snapshot["original_root"],new_copy=str(root),changed_existing=sorted(changed),
        added=sorted(paths.keys()-baseline.keys()),removed_or_archived=sorted(baseline.keys()-paths.keys()))
    for name in ("change_manifest.json","changed_existing_files.json"):
        (root/"docs"/name).write_text(json.dumps(report if name.startswith("change_manifest") else sorted(changed),
            ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(changed_existing=len(changed),added=len(report["added"]),archived=len(report["removed_or_archived"]))))

if __name__=="__main__": main()
