"""제출용 submission.zip 생성 스크립트.

사용법: 이 파일을 myalgorithm.py / utils.py / baseline_greedy.py 가 있는
폴더에 넣고 VS Code 터미널에서 실행:

    python make_zip.py

같은 폴더에 submission.zip 이 만들어짐 (파일 3개가 zip 루트에 위치).
"""
import pathlib
import sys
import zipfile

REQUIRED = ["myalgorithm.py", "utils.py", "baseline_greedy.py"]


def main():
    here = pathlib.Path(__file__).parent
    missing = [f for f in REQUIRED if not (here / f).exists()]
    if missing:
        print(f"실패: 이 폴더에 다음 파일이 없음 -> {missing}")
        print(f"현재 폴더: {here}")
        sys.exit(1)

    out = here / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in REQUIRED:
            z.write(here / f, arcname=f)   # arcname=f -> 폴더 없이 루트에 저장

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"생성 완료: {out}")
    print(f"zip 내용: {names}")
    assert set(names) == set(REQUIRED), "zip 내용이 예상과 다름!"
    print("OK -- 이제 check_zip.py 로 검증하세요:")
    print(r"  python check_zip.py submission.zip 예제경로\example_B2_b10.json 30")


if __name__ == "__main__":
    main()
