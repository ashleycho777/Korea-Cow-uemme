#!/usr/bin/env python3
"""
OGC 2026 평가 하네스 (evaluation harness)
=========================================

내 알고리즘이 얼마나 잘 푸는지 진단하는 도구.
- 실행 시간 측정 (시간제한 대비 초과 여부)
- check_feasibility() 로 실행가능성 / 실패 단계 확인
- 목적함수 지각(Z1)·불균형(Z2)·선호(Z3)의 *가중 기여도*를 비중(%)으로 분해
  -> "지금 어디서 점수를 잃고 있는가"가 바로 보임
- 여러 인스턴스 일괄 실행 + 요약표
- (옵션) 서브프로세스 격리로 크래시/시간초과까지 서버처럼 잡기
- (옵션) 결과를 jsonl 로그에 append -> 개선 추이 추적

사용 예시
---------
  # 기본: myalgorithm.algorithm 을 example_B2_b10.json 에 돌림
  python eval_harness.py example_B2_b10.json

  # 여러 인스턴스 (글롭 사용 가능)
  python eval_harness.py "instances/*.json" --timelimit 60

  # 다른 알고리즘 모듈/함수 지정
  python eval_harness.py prob.json --algo baseline_greedy:greedyalgorithm

  # 알고리즘 stdout 까지 보기
  python eval_harness.py prob.json --verbose

  # 서버처럼 격리 실행 (타임아웃 강제종료 + 크래시 감지)
  python eval_harness.py prob.json --isolate --timelimit 60

  # 결과를 로그에 남기기 (개선 추적용)
  python eval_harness.py "*.json" --log runs.jsonl --note "greedy baseline"

알고리즘 파일(myalgorithm.py)과 utils.py 가 같은 폴더에서 실행하거나,
--algo-dir 로 그 폴더를 지정하세요.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

# Windows 콘솔(cp949/cp1252 등)에서 박스 문자(█)·한글 출력이 깨지거나
# UnicodeEncodeError 로 죽는 것을 방지 -> stdout/stderr 를 UTF-8 로 강제.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 결과 상태 (서버 응답 상태를 흉내낸다)
# ---------------------------------------------------------------------------
ST_FEASIBLE = "FEASIBLE"        # 실행가능한 해를 찾음
ST_INFEASIBLE = "INFEASIBLE"    # check_feasibility 실패
ST_EXCEPTION = "EXCEPTION"      # 알고리즘이 예외를 던짐
ST_CRASHED = "CRASHED"          # 프로세스가 죽음 (격리 모드에서만)
ST_TIMEOUT = "TIMEOUT"          # 시간제한 초과로 강제 종료 (격리 모드에서만)
ST_BAD_OUTPUT = "BAD_OUTPUT"    # 해 형식이 잘못됨


# ---------------------------------------------------------------------------
# 알고리즘 / utils 로딩
# ---------------------------------------------------------------------------
def _prepare_path(algo_dir: str) -> None:
    """알고리즘과 utils 를 import 할 수 있도록 폴더를 sys.path 맨 앞에 추가."""
    abs_dir = os.path.abspath(algo_dir)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)


def load_algorithm(spec: str):
    """'module:function' 또는 'module' (이 경우 함수명은 algorithm) 을 불러옴."""
    if ":" in spec:
        mod_name, fn_name = spec.split(":", 1)
    else:
        mod_name, fn_name = spec, "algorithm"
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)  # 코드 수정 후 재실행 시 최신 반영
    return getattr(mod, fn_name)


def load_utils():
    import utils  # noqa: WPS433 (project utils.py)
    return utils


# ---------------------------------------------------------------------------
# 인스턴스 로딩
# ---------------------------------------------------------------------------
def discover_instances(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(pat))
        if matched:
            paths.extend(matched)
        elif os.path.exists(pat):
            paths.append(pat)
        else:
            print(f"  [warn] 일치하는 인스턴스 없음: {pat}", file=sys.stderr)
    # 중복 제거, 순서 유지
    seen, out = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(p)
    return out


def load_instance(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 단일 실행 (인-프로세스)
# ---------------------------------------------------------------------------
def run_one_inprocess(algo_fn, utils, prob_info: dict, timelimit: float,
                      capture: bool = True) -> dict:
    """알고리즘을 직접 호출하고 측정. 빠르고 트레이스백을 볼 수 있음."""
    buf = io.StringIO()
    err = io.StringIO()
    status = ST_FEASIBLE
    solution = None
    tb = None

    t0 = time.perf_counter()
    try:
        if capture:
            with redirect_stdout(buf), redirect_stderr(err):
                solution = algo_fn(prob_info, timelimit)
        else:
            solution = algo_fn(prob_info, timelimit)
    except Exception:
        status = ST_EXCEPTION
        tb = traceback.format_exc()
    elapsed = time.perf_counter() - t0

    result = {
        "status": status,
        "elapsed": elapsed,
        "traceback": tb,
        "algo_stdout": buf.getvalue() if capture else "",
    }

    if status == ST_EXCEPTION:
        return result

    # 형식 사전 점검 (check_feasibility 도 점검하지만 더 친절한 메시지를 위해)
    if not isinstance(solution, dict) or "operations" not in solution:
        result["status"] = ST_BAD_OUTPUT
        result["violations"] = ["해는 'operations' 키를 가진 dict 여야 합니다."]
        return result

    fr = utils.check_feasibility(prob_info, solution)
    result["feasible"] = fr.get("feasible", False)
    result["stage"] = fr.get("stage")
    result["violations"] = fr.get("violations", [])
    result["objective"] = fr.get("objective")
    result["obj1"] = fr.get("obj1")
    result["obj2"] = fr.get("obj2")
    result["obj3"] = fr.get("obj3")
    if not result["feasible"]:
        result["status"] = ST_INFEASIBLE
    return result


# ---------------------------------------------------------------------------
# 단일 실행 (서브프로세스 격리 — 타임아웃/크래시 감지)
# ---------------------------------------------------------------------------
def _worker(algo_dir, algo_spec, prob_info, timelimit, q):
    try:
        _prepare_path(algo_dir)
        utils = load_utils()
        algo_fn = load_algorithm(algo_spec)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            sol = algo_fn(prob_info, timelimit)
        if not isinstance(sol, dict) or "operations" not in sol:
            q.put({"status": ST_BAD_OUTPUT,
                   "violations": ["해는 'operations' 키를 가진 dict 여야 합니다."],
                   "algo_stdout": buf.getvalue()})
            return
        fr = utils.check_feasibility(prob_info, sol)
        q.put({
            "status": ST_FEASIBLE if fr.get("feasible") else ST_INFEASIBLE,
            "feasible": fr.get("feasible", False),
            "stage": fr.get("stage"),
            "violations": fr.get("violations", []),
            "objective": fr.get("objective"),
            "obj1": fr.get("obj1"), "obj2": fr.get("obj2"), "obj3": fr.get("obj3"),
            "algo_stdout": buf.getvalue(),
        })
    except Exception:
        q.put({"status": ST_EXCEPTION, "traceback": traceback.format_exc()})


def run_one_isolated(algo_dir, algo_spec, prob_info, timelimit,
                     kill_grace: float = 5.0) -> dict:
    """서브프로세스에서 실행. 시간제한 초과 시 강제 종료, 시그널핸들 등도 감지."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker,
                    args=(algo_dir, algo_spec, prob_info, timelimit, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timelimit + kill_grace)
    elapsed = time.perf_counter() - t0

    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": ST_TIMEOUT, "elapsed": elapsed,
                "violations": [f"시간제한 {timelimit}s + 여유 {kill_grace}s 초과로 강제 종료"]}
    if not q.empty():
        res = q.get()
        res["elapsed"] = elapsed
        return res
    # 큐가 비었는데 프로세스가 끝남 -> 크래시(시그널핸들/포크 등)
    return {"status": ST_CRASHED, "elapsed": elapsed,
            "violations": [f"프로세스가 비정상 종료 (exitcode={p.exitcode})"]}


# ---------------------------------------------------------------------------
# 리포트 출력
# ---------------------------------------------------------------------------
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{_C.get(color, '')}{text}{_C['reset']}"


def _status_str(status: str) -> str:
    color = {
        ST_FEASIBLE: "green", ST_INFEASIBLE: "red", ST_EXCEPTION: "red",
        ST_CRASHED: "red", ST_TIMEOUT: "yellow", ST_BAD_OUTPUT: "red",
    }.get(status, "yellow")
    return _c(status, color)


def print_report(name: str, prob_info: dict, res: dict, timelimit: float,
                 verbose: bool) -> None:
    weights = prob_info.get("weights", {})
    n_bays = len(prob_info.get("bays", []))
    n_blocks = len(prob_info.get("blocks", []))

    print(_c("=" * 64, "dim"))
    print(f"{_c(name, 'bold')}   "
          f"({_c(str(n_bays), 'cyan')} bays, {_c(str(n_blocks), 'cyan')} blocks)")

    elapsed = res.get("elapsed", 0.0)
    over = elapsed > timelimit
    time_str = f"{elapsed:.3f}s / {timelimit:.0f}s"
    if over:
        time_str = _c(time_str + "  ← 시간제한 초과", "red")
    print(f"  status : {_status_str(res['status'])}    "
          f"time : {time_str}")

    if verbose and res.get("algo_stdout"):
        print(_c("  --- 알고리즘 stdout ---", "dim"))
        for line in res["algo_stdout"].rstrip().splitlines():
            print(_c("  | " + line, "dim"))

    if res["status"] in (ST_EXCEPTION,):
        print(_c("  예외 발생:", "red"))
        for line in (res.get("traceback") or "").rstrip().splitlines():
            print("    " + line)
        return

    if res["status"] in (ST_INFEASIBLE, ST_BAD_OUTPUT, ST_CRASHED, ST_TIMEOUT):
        if res.get("stage") is not None:
            print(f"  실패 단계 (stage) : {_c(str(res['stage']), 'red')}")
        viols = res.get("violations", [])
        if viols:
            print(f"  위반 {len(viols)}건 (첫 5건):")
            for v in viols[:5]:
                print(_c("    - " + str(v), "red"))
        return

    # ---- FEASIBLE: 목적함수 분해 ----
    w1 = weights.get("w1", 0); w2 = weights.get("w2", 0); w3 = weights.get("w3", 0)
    o1 = res["obj1"] or 0; o2 = res["obj2"] or 0; o3 = res["obj3"] or 0
    c1, c2, c3 = w1 * o1, w2 * o2, w3 * o3
    total = res["objective"] or (c1 + c2 + c3)

    print(f"  {_c('objective', 'bold')} = {_c(f'{total:.0f}', 'green')}")
    print(f"    {'항목':<22}{'raw':>10}{'×weight':>10}{'기여':>12}{'비중':>8}")
    rows = [
        ("Z1 지각 (tardiness)", o1, w1, c1),
        ("Z2 불균형 (imbalance)", o2, w2, c2),
        ("Z3 선호 (preference)", o3, w3, c3),
    ]
    for label, raw, w, contrib in rows:
        pct = (contrib / total * 100) if total else 0.0
        bar = _bar(pct)
        print(f"    {label:<22}{raw:>10.1f}{w:>10}{contrib:>12.0f}"
              f"{pct:>7.1f}% {bar}")


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return _c("█" * filled + "·" * (width - filled), "cyan")


def print_summary(results: list[tuple[str, dict]]) -> None:
    print(_c("=" * 64, "dim"))
    print(_c("요약", "bold"))
    feas = sum(1 for _, r in results if r["status"] == ST_FEASIBLE)
    print(f"  실행가능 : {feas}/{len(results)}")
    print(f"  {'instance':<26}{'status':<12}{'objective':>12}{'time':>10}")
    for name, r in results:
        obj = r.get("objective")
        obj_str = f"{obj:.0f}" if obj is not None else "-"
        st = r["status"]
        st_disp = st if not _use_color() else _status_str(st)
        # 색상 코드 길이 보정을 위해 raw 길이 기준 패딩
        pad = 12 - len(st)
        print(f"  {os.path.basename(name):<26}{st_disp}{' ' * max(pad,1)}"
              f"{obj_str:>12}{r.get('elapsed',0):>9.2f}s")


# ---------------------------------------------------------------------------
# 로그 (개선 추적용)
# ---------------------------------------------------------------------------
def append_log(log_path: str, name: str, res: dict, timelimit: float,
               note: str) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "instance": os.path.basename(name),
        "status": res["status"],
        "objective": res.get("objective"),
        "obj1": res.get("obj1"), "obj2": res.get("obj2"), "obj3": res.get("obj3"),
        "elapsed": round(res.get("elapsed", 0.0), 4),
        "timelimit": timelimit,
        "note": note,
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="OGC 2026 알고리즘 평가 하네스",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="+",
                    help="인스턴스 JSON 경로 (글롭 가능, 예: 'inst/*.json')")
    ap.add_argument("--algo", default="myalgorithm",
                    help="알고리즘 'module' 또는 'module:function' (기본 myalgorithm)")
    ap.add_argument("--algo-dir", default=".",
                    help="myalgorithm.py / utils.py 가 있는 폴더 (기본 현재 폴더)")
    ap.add_argument("--timelimit", type=float, default=60.0,
                    help="알고리즘 시간제한(초) (기본 60)")
    ap.add_argument("--isolate", action="store_true",
                    help="서브프로세스 격리: 타임아웃/크래시 감지")
    ap.add_argument("--repeat", type=int, default=1,
                    help="각 인스턴스 N회 반복 후 최선 결과 보고 (타이밍 변동 확인)")
    ap.add_argument("--verbose", action="store_true",
                    help="알고리즘 stdout 표시")
    ap.add_argument("--log", default=None,
                    help="결과를 jsonl 로 append (개선 추적)")
    ap.add_argument("--note", default="",
                    help="--log 와 함께 기록할 메모")
    args = ap.parse_args(argv)

    _prepare_path(args.algo_dir)
    try:
        utils = load_utils()
    except Exception as exc:  # noqa: BLE001
        print(_c(f"[error] utils.py 를 불러올 수 없음 ({exc}). "
                 f"--algo-dir 를 확인하세요.", "red"), file=sys.stderr)
        return 2

    algo_fn = None
    if not args.isolate:
        try:
            algo_fn = load_algorithm(args.algo)
        except Exception as exc:  # noqa: BLE001
            print(_c(f"[error] 알고리즘 '{args.algo}' 로드 실패: {exc}", "red"),
                  file=sys.stderr)
            return 2

    inst_paths = discover_instances(args.instances)
    if not inst_paths:
        print(_c("[error] 실행할 인스턴스가 없습니다.", "red"), file=sys.stderr)
        return 2

    summary: list[tuple[str, dict]] = []
    for path in inst_paths:
        prob_info = load_instance(path)
        name = prob_info.get("name", os.path.basename(path))

        best = None
        for _ in range(max(1, args.repeat)):
            if args.isolate:
                res = run_one_isolated(args.algo_dir, args.algo,
                                       prob_info, args.timelimit)
            else:
                res = run_one_inprocess(algo_fn, utils, prob_info,
                                        args.timelimit,
                                        capture=not args.verbose)
            # 최선 = feasible 우선, 그 안에서 objective 최소
            if best is None or _better(res, best):
                best = res

        print_report(name, prob_info, best, args.timelimit, args.verbose)
        summary.append((name, best))
        if args.log:
            append_log(args.log, name, best, args.timelimit, args.note)

    if len(summary) > 1:
        print_summary(summary)

    # 종료 코드: 하나라도 실행불가능하면 1
    all_feasible = all(r["status"] == ST_FEASIBLE for _, r in summary)
    return 0 if all_feasible else 1


def _better(a: dict, b: dict) -> bool:
    """a 가 b 보다 더 나은 결과인가? (feasible 우선, objective 최소)"""
    af = a["status"] == ST_FEASIBLE
    bf = b["status"] == ST_FEASIBLE
    if af != bf:
        return af
    if af and bf:
        return (a.get("objective", float("inf")) <
                b.get("objective", float("inf")))
    return False


if __name__ == "__main__":
    raise SystemExit(main())
