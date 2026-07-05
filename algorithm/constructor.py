"""
constructor.py — 구성 휴리스틱 + 상태 API (2단계 모듈 B, 상태 기반 리팩터)
========================================================================

placement.py(모듈 A) 를 시간 축 위에서 여러 번 눌러서, 다섯 결정
(베이·위치·방향·지연·취소)을 채우도록 시킨다.

이 버전은 LNS(3단계)가 쓸 수 있도록 '상태(State)' API 를 노출한다:
  - new_state(prob_info)          : 빈 상태
  - insert_block(state, bid, ...) : 블록 1개를 최적 베이·시간·위치에 배치
  - remove_block(state, bid)      : 블록 1개 제거
  - clone_state(state)            : 상태 복사(되돌리기용)
  - objective(state)              : 채점 기준과 동일한 (obj, o1, o2, o3)
  - build_operations(state)       : 최종 오퍼레이션 딕셔너리

흐름/실행가능성/안전 여지 설명은 기존과 동일(README 주석 참조).

공개 진입점:  solve(prob_info, timelimit=60) -> {"operations": {...}}
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

from utils import Block, Bay
import placement as P


# ---------------------------------------------------------------------------
# 자료구조
# ---------------------------------------------------------------------------
@dataclass
class PlacedBlock:
    block: Block
    entry: int
    exit: int
    workload: float
    bay_id: int
    obstacle: P.Obstacle


@dataclass
class State:
    instance: dict
    bays: list
    w1: float
    w2: float
    w3: float
    u: list                      # 베이 가중치
    placed: dict                 # block_id -> PlacedBlock
    by_bay: list                 # bay_id -> list[PlacedBlock]
    load: list                   # bay_id -> 누적 workload


# ---------------------------------------------------------------------------
# 보조 계산
# ---------------------------------------------------------------------------
def _bay_weights(bays) -> list:
    areas = [b.width * b.height for b in bays]
    avg = sum(areas) / len(areas)
    return [avg / a for a in areas]


def _block_area(block_data: dict) -> float:
    layers = block_data["shape"][0]["layers"]
    xs = [v[0] for layer in layers for v in layer]
    ys = [v[1] for layer in layers for v in layer]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs else 0.0


def order_blocks_edd(blocks: list) -> list:
    """EDD: 마감 asc, 릴리즈 asc, 면적 큰 것 먼저."""
    idx = list(range(len(blocks)))
    idx.sort(key=lambda i: (blocks[i]["due_date"],
                            blocks[i]["release_time"],
                            -_block_area(blocks[i])))
    return idx


def _candidate_times(placed: list, release: int, proc: int) -> list:
    cand = {release}
    for pb in placed:
        if pb.exit >= release:
            cand.add(pb.exit)
        e = pb.entry - proc
        if e >= release:
            cand.add(e)
    if placed:
        cand.add(max(release, max(pb.exit for pb in placed)))
    return sorted(cand)


def _obstacles_at(placed: list, t: int, proc: int) -> list:
    end = t + proc
    return [pb.obstacle for pb in placed if pb.entry < end and t < pb.exit]


def _earliest_in_bay(bay, bid, instance, placed, release, proc, deadline):
    times = _candidate_times(placed, release, proc)
    for t in times:
        if deadline is not None and time.perf_counter() > deadline:
            # 시간 초과: '빈-베이 보장 시간'(후보 중 최댓값)만 시도.
            # 이 시간엔 장애물이 없으므로, 블록이 이 베이에 들어가면 반드시 배치된다.
            t_empty = times[-1]
            pl = P.find_placement(bay, bid, instance,
                                  _obstacles_at(placed, t_empty, proc))
            return (t_empty, pl) if pl is not None else None
        pl = P.find_placement(bay, bid, instance, _obstacles_at(placed, t, proc))
        if pl is not None:
            return t, pl
    return None


# ---------------------------------------------------------------------------
# 상태 API
# ---------------------------------------------------------------------------
def new_state(prob_info: dict) -> State:
    bays = [Bay.from_dict(b, i) for i, b in enumerate(prob_info["bays"])]
    w = prob_info.get("weights", {})
    return State(
        instance=prob_info, bays=bays,
        w1=w.get("w1", 1.0), w2=w.get("w2", 0.0), w3=w.get("w3", 0.0),
        u=_bay_weights(bays),
        placed={}, by_bay=[[] for _ in bays], load=[0.0] * len(bays),
    )


def clone_state(s: State) -> State:
    return State(
        instance=s.instance, bays=s.bays, w1=s.w1, w2=s.w2, w3=s.w3, u=s.u,
        placed=dict(s.placed),
        by_bay=[lst[:] for lst in s.by_bay],
        load=s.load[:],
    )


def insert_block(state: State, bid: int, deadline=None) -> bool:
    """블록 bid 를 현재 상태의 최적(목적함수 미러링)으로 배치. 성공 여부 반환."""
    bd = state.instance["blocks"][bid]
    R = bd["release_time"]; D = bd["due_date"]
    P_ = bd["processing_time"]; L = bd["workload"]
    prefs = bd["bay_preferences"]; s_max = max(prefs)

    best = None  # (score, bay_id, entry, placement)
    for j, bay in enumerate(state.bays):
        res = _earliest_in_bay(bay, bid, state.instance, state.by_bay[j],
                               R, P_, deadline)
        if res is None:
            continue
        entry, pl = res
        exit_ = entry + P_
        tard = max(0, exit_ - D)
        pref_pen = s_max - prefs[j]
        new_load = state.load[j] + L
        bal = max((abs(state.u[j] * new_load - state.u[k] * state.load[k])
                   for k in range(len(state.bays)) if k != j), default=0.0)
        score = (state.w1 * tard + state.w2 * bal + state.w3 * pref_pen
                 + 1e-4 * pl.top_y)
        if best is None or score < best[0]:
            best = (score, j, entry, pl)

    if best is None:
        return False
    _, j, entry, pl = best
    exit_ = entry + P_
    pb = PlacedBlock(block=pl.block, entry=entry, exit=exit_, workload=L,
                     bay_id=j, obstacle=P.make_obstacle(pl.block))
    state.placed[bid] = pb
    state.by_bay[j].append(pb)
    state.load[j] += L
    return True


def remove_block(state: State, bid: int) -> None:
    pb = state.placed.pop(bid, None)
    if pb is None:
        return
    state.by_bay[pb.bay_id].remove(pb)
    state.load[pb.bay_id] -= pb.workload


def objective(state: State):
    """채점 기준과 동일한 (objective, obj1, obj2, obj3)."""
    blocks = state.instance["blocks"]
    o1 = 0.0; o3 = 0.0
    for bid, pb in state.placed.items():
        blk = blocks[bid]
        o1 += max(0.0, pb.exit - blk["due_date"])
        o3 += max(blk["bay_preferences"]) - blk["bay_preferences"][pb.bay_id]
    n = len(state.bays)
    if n >= 2:
        o2 = math.floor(max(
            abs(state.u[a] * state.load[a] - state.u[b] * state.load[b])
            for a in range(n) for b in range(n) if a != b))
    else:
        o2 = 0.0
    obj = state.w1 * o1 + state.w2 * o2 + state.w3 * o3
    return obj, o1, o2, o3


def build_operations(state: State) -> dict:
    entries = defaultdict(list); exits = defaultdict(list)
    for bid, pb in state.placed.items():
        entries[pb.entry].append({
            "type": "ENTRY", "block_id": pb.block.block_id, "bay_id": pb.bay_id,
            "x": int(pb.block.x), "y": int(pb.block.y),
            "orient_idx": int(pb.block.orient_idx),
        })
        exits[pb.exit].append({
            "type": "EXIT", "block_id": pb.block.block_id, "bay_id": pb.bay_id,
        })
    operations = {}
    for t in sorted(set(entries) | set(exits)):
        operations[str(int(t))] = exits.get(t, []) + entries.get(t, [])
    return {"operations": operations}


# ---------------------------------------------------------------------------
# 그리드 구성
# ---------------------------------------------------------------------------
def construct(prob_info: dict, timelimit: float = 60.0,
              verbose: bool = False) -> dict:
    t0 = time.perf_counter()
    deadline = t0 + max(1.0, timelimit - 1.0)
    state = build_state(prob_info, deadline, verbose)
    if verbose:
        obj, o1, o2, o3 = objective(state)
        print(f"[constructor] obj={obj:.0f} (Z1={o1:.0f} Z2={o2:.0f} Z3={o3:.0f}) "
              f"in {time.perf_counter()-t0:.2f}s")
    return build_operations(state)


def build_state(prob_info: dict, deadline=None, verbose: bool = False) -> State:
    """그리드로 채운 State 를 반환(LNS 초기화에도 사용)."""
    state = new_state(prob_info)
    for bid in order_blocks_edd(prob_info["blocks"]):
        ok = insert_block(state, bid, deadline)
        if not ok and verbose:
            print(f"[constructor] WARNING: block {bid} 배치 불가")
    return state


def solve(prob_info: dict, timelimit: float = 60.0) -> dict:
    P.clear_cache()          # 인스턴스별 기하 캐시 격리
    return construct(prob_info, timelimit=timelimit, verbose=False)


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "example_B2_b10.json"
    inst = json.load(open(path, encoding="utf-8"))
    construct(inst, timelimit=30, verbose=True)
