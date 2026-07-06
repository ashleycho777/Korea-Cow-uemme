"""
make_submission.py — OGC 2026 제출 zip 생성기
==============================================

저장소 루트에서 실행하면, 대회 규칙(최상위에 myalgorithm.py, 필요한 파일 전부 포함)에
맞춰 제출용 zip 을 만든다. 매일 재제출 전에 한 번 돌리면 됨.

    python make_submission.py

옵션:
    python make_submission.py --out my_entry.zip
"""
import argparse
import os
import sys
import zipfile

# 제출에 반드시 포함할 파일 (모두 저장소 루트에 있어야 함)
REQUIRED = ["myalgorithm.py", "placement.py", "constructor.py", "lns.py"]
# 있으면 포함(대회 제공본/설명). utils.py 는 서버가 덮어쓰지만 자립성을 위해 포함.
OPTIONAL = ["utils.py", "README.txt"]


def main() -> int:
    ap = argparse.ArgumentParser(description="OGC 2026 제출 zip 생성")
    ap.add_argument("--out", default="ogc2026_submission.zip", help="출력 zip 경로")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        print("[오류] 필수 파일이 없습니다:", ", ".join(missing))
        print("      이 스크립트를 저장소 루트(myalgorithm.py 가 있는 폴더)에서 실행하세요.")
        return 1

    files = REQUIRED + [f for f in OPTIONAL if os.path.exists(f)]
    if os.path.exists(args.out):
        os.remove(args.out)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))  # 항상 최상위에 배치

    print(f"[완료] {args.out} 생성")
    with zipfile.ZipFile(args.out) as z:
        total = 0
        for info in z.infolist():
            total += info.file_size
            print(f"   - {info.filename} ({info.file_size:,} bytes)")
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"   압축 크기: {size_mb:.2f} MB (제한 15MB)")
    if "utils.py" not in files:
        print("   참고: utils.py 미포함 — 서버 제공본을 사용합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
