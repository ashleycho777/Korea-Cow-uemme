"""제출 zip 검증 스크립트.

사용법:
    python check_zip.py submission.zip 예제경로\\example_B2_b10.json [타임리밋]

하는 일:
  1. zip 내용 확인 (myalgorithm.py / utils.py / baseline_greedy.py 가 루트에 있는지)
  2. zip 을 임시 폴더에 풀고, 서버처럼 거기서 algorithm() 실행
  3. check_feasibility 로 feasible 여부 + objective 출력
"""
import importlib
import json
import pathlib
import sys
import tempfile
import time
import zipfile

REQUIRED = {"myalgorithm.py", "utils.py", "baseline_greedy.py"}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    zip_path = pathlib.Path(sys.argv[1])
    inst_path = pathlib.Path(sys.argv[2])
    timelimit = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    # -- 1. zip 내용 확인 -----------------------------------------------------
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    print(f"[1/3] zip 내용: {names}")
    missing = REQUIRED - set(names)
    nested = [n for n in names if "/" in n or "\\" in n]
    if missing:
        print(f"  !! 실패: 필수 파일 누락 {missing}")
        sys.exit(1)
    if nested:
        print(f"  !! 경고: 파일이 폴더 안에 들어있음 {nested}")
        print("     -> 파일 3개를 '직접' 선택해서 압축해야 함 (폴더째 압축 X)")
        sys.exit(1)
    print("  OK: 3개 파일이 zip 루트에 존재")

    # -- 2. 임시 폴더에 풀고 실행 ---------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        sys.path.insert(0, tmp)
        # 현재 폴더에 같은 이름 모듈이 있어도 zip 쪽이 우선되도록 캐시 제거
        for m in ("myalgorithm", "utils", "baseline_greedy"):
            sys.modules.pop(m, None)
        myalg = importlib.import_module("myalgorithm")
        utils = importlib.import_module("utils")

        prob = json.loads(inst_path.read_text(encoding="utf-8"))
        print(f"[2/3] algorithm() 실행 (timelimit={timelimit:.0f}s) ...")
        t0 = time.time()
        sol = myalg.algorithm(prob, timelimit=timelimit)
        elapsed = time.time() - t0

        # -- 3. feasibility 확인 ---------------------------------------------
        r = utils.check_feasibility(prob, sol)
        print(f"[3/3] 결과: elapsed={elapsed:.1f}s (제한 {timelimit:.0f}s)")
        if r["feasible"]:
            print(f"  ✔ FEASIBLE  objective={r['objective']:.0f}"
                  f"  (obj1={r['obj1']:.0f} obj2={r['obj2']:.0f} obj3={r['obj3']:.0f})")
            if elapsed > timelimit:
                print("  !! 경고: 시간 초과 -- 서버에서 실격될 수 있음")
                sys.exit(1)
            print("\n제출해도 됩니다.")
        else:
            print(f"  ✘ INFEASIBLE stage={r['stage']}")
            for v in r.get("violations", [])[:5]:
                print(f"    {v}")
            sys.exit(1)


if __name__ == "__main__":
    main()
