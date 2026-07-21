"""
lns.py — Large Neighborhood Search 개선 루프 (3단계 모듈 D)
==========================================================

constructor 의 그리디 해에서 출발해, 남는 시간제한 동안
'일부를 뜯어내고(destroy) 다시 끼워넣기(repair)'를 반복하며 점수를 낮춘다.
시간을 쓸수록 좋아지는 anytime 알고리즘.

왜 항상 실현가능한가
--------------------
repair 는 constructor.insert_block 을 쓰고, 그 후보 시각에는 '빈-베이 시각'이
항상 포함된다 -> 어떤 블록도 (자기가 들어갈 베이가 있으면) 반드시 재배치된다.
게다가 안전영역(footprint 비중첩)이라 충돌·크레인도 자동 통과.

구성
----
- destroy: worst_tardiness / random / related(Shaw)   (적응형 가중 선택)
- repair : 제거 블록을 (여유 적은 순 / EDD / 무작위) 로 재삽입
- accept : Record-to-Record Travel (최고해 대비 허용폭이 시간에 따라 수축)

공개 진입점:  solve(prob_info, timelimit=60) -> {"operations": {...}}
"""
from __future__ import annotations

import math
import random
import time

import constructor as C
import placement as P


# ---------------------------------------------------------------------------
# Destroy 연산자들  (state 를 직접 변형하고, 제거한 block_id 리스트를 반환)
# ---------------------------------------------------------------------------
def destroy_worst_tardiness(state: C.State, k: int, rng: random.Random) -> list:
    """지각이 큰 블록 우선 제거. 부족하면 무작위로 채움."""
    blocks = state.instance["blocks"]
    scored = []
    for bid, pb in state.placed.items():
        tard = max(0, pb.exit - blocks[bid]["due_date"])
        scored.append((tard, bid))
    scored.sort(reverse=True)
    chosen = [bid for tard, bid in scored if tard > 0][:k]
    if len(chosen) < k:
        rest = [bid for _, bid in scored if bid not in chosen]
        rng.shuffle(rest)
        chosen += rest[:k - len(chosen)]
    for bid in chosen:
        C.remove_block(state, bid)
    return chosen


def destroy_random(state: C.State, k: int, rng: random.Random) -> list:
    bids = list(state.placed.keys())
    rng.shuffle(bids)
    chosen = bids[:k]
    for bid in chosen:
        C.remove_block(state, bid)
    return chosen


def _centroid(pb: C.PlacedBlock):
    x0, y0, x1, y1 = pb.obstacle.bbox
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def destroy_related(state: C.State, k: int, rng: random.Random) -> list:
    """Shaw 제거: 씨앗 블록과 '연관'(같은 베이·시간겹침·공간근접)된 블록들을 함께 제거."""
    bids = list(state.placed.keys())
    if not bids:
        return []
    seed = rng.choice(bids)
    sp = state.placed[seed]
    sx, sy = _centroid(sp)

    def relatedness(bid):
        pb = state.placed[bid]
        same_bay = 1.0 if pb.bay_id == sp.bay_id else 0.0
        # 시간 겹침 여부
        overlap = 1.0 if (pb.entry < sp.exit and sp.entry < pb.exit) else 0.0
        cx, cy = _centroid(pb)
        dist = math.hypot(cx - sx, cy - sy)
        # 클수록 더 연관 (가까울수록, 같은 베이/시간겹침일수록)
        return same_bay + overlap + 1.0 / (1.0 + dist)

    others = [b for b in bids if b != seed]
    others.sort(key=relatedness, reverse=True)
    chosen = [seed] + others[:max(0, k - 1)]
    for bid in chosen:
        C.remove_block(state, bid)
    return chosen


def destroy_tardy_cluster(state: C.State, k: int, rng: random.Random) -> list:
    """가장 지각 큰 블록 + 그 블록의 이른 배치를 막는(같은 베이·시간창 겹침) 블록들을 함께 제거.
    자리를 터줘서 repair 가 지각 블록을 일찍 넣을 수 있게 한다. 지각(Z1)을 직접 겨냥."""
    blocks = state.instance["blocks"]
    tardy = [(max(0, pb.exit - blocks[bid]["due_date"]), bid)
             for bid, pb in state.placed.items()]
    tardy.sort(reverse=True)
    if not tardy or tardy[0][0] == 0:
        return destroy_random(state, k, rng)   # 지각 없으면 무작위로
    seed = tardy[0][1]
    sp = state.placed[seed]
    R = blocks[seed]["release_time"]
    # 같은 베이에서, 씨앗이 일찍 있고 싶은 창 [R, exit) 와 시간 겹치는 블록들(자리 점유 후보)
    conflict = [bid for bid, pb in state.placed.items()
                if bid != seed and pb.bay_id == sp.bay_id
                and pb.entry < sp.exit and R < pb.exit]
    rng.shuffle(conflict)
    chosen = [seed] + conflict[:max(0, k - 1)]
    for bid in chosen:
        C.remove_block(state, bid)
    return chosen


DESTROY_OPS = [destroy_worst_tardiness, destroy_random, destroy_related,
               destroy_tardy_cluster]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
def repair(state: C.State, removed: list, rng: random.Random,
           deadline: float) -> None:
    blocks = state.instance["blocks"]
    mode = rng.random()
    if mode < 0.5:        # 여유(slack) 적은 순 — 지각 위험 큰 것 먼저
        removed = sorted(removed, key=lambda b: (
            blocks[b]["due_date"] - blocks[b]["release_time"]
            - blocks[b]["processing_time"]))
    elif mode < 0.8:      # EDD
        removed = sorted(removed, key=lambda b: blocks[b]["due_date"])
    else:                 # 무작위 (다양화)
        removed = removed[:]
        rng.shuffle(removed)
    for bid in removed:
        C.insert_block(state, bid, deadline)


# ---------------------------------------------------------------------------
# 메인 LNS 루프
# ---------------------------------------------------------------------------
def lns(prob_info: dict, timelimit: float = 60.0, seed: int = 12345,
        verbose: bool = False, force_fast: bool = False) -> dict:
    t0 = time.perf_counter()
    safety = min(3.0, 0.05 * timelimit)
    deadline = t0 + max(1.0, timelimit - safety)

    # --- 시작점: 그리디 구성 ---
    # 폴리곤(고밀도)으로 풀되, 예산의 40%를 넘기면 남은 블록만 AABB 고속으로 전환.
    # 보통 인스턴스(P4~P6 포함)는 그 전에 끝나 폴리곤 품질 그대로. 거대 인스턴스만 폴백.
    # force_fast=True 면 처음부터 AABB(실험/거대인스턴스용).
    # 전환점 70%: 폴리곤 구성이 거의 항상 먼저 끝나 순수 폴리곤 출발점을 보장.
    # (정말 거대해 폴리곤이 못 끝나는 경우에만 AABB 폴백 -> −1 방지.)
    fast_switch = t0 if force_fast else t0 + max(2.0, 0.7 * (deadline - t0))
    current = C.build_state(prob_info, deadline=deadline, verbose=verbose,
                            fast_switch=fast_switch)
    # 재삽입(repair) 엔진: 구성은 폴리곤(고밀도)로 끝냈고, LNS 반복은 고속 격자로 전환한다.
    # -> 밀도 높은 출발점 + 수천 번의 빠른 LNS 반복. 최선해 보관이라 절대 나빠지지 않음.
    # 단, 거대 인스턴스 폴백('aabb')이 걸렸으면 그대로 aabb 유지.
    if current.place_mode == "poly":
        current.place_mode = "grid"
    best = C.clone_state(current)
    best_obj = C.objective(current)[0]
    cur_obj = best_obj

    n = len(prob_info["blocks"])
    kmax = max(3, min(12, int(0.15 * n) + 2))

    # 적응형 destroy 가중치
    weights = [1.0] * len(DESTROY_OPS)
    decay = 0.9

    iters = 0; accepts = 0; improves = 0
    if verbose:
        print(f"[lns] start obj={best_obj:.0f}  (construct {time.perf_counter()-t0:.1f}s, "
              f"budget {deadline-t0:.1f}s)")

    while time.perf_counter() < deadline and n > 1:
        iters += 1
        progress = (time.perf_counter() - t0) / (deadline - t0)
        dev = 0.02 * (1.0 - progress)              # RRT 허용폭, 0 으로 수축

        cand = C.clone_state(current)
        k = rng_k(weights, kmax)
        op_idx = roulette(weights)
        removed = DESTROY_OPS[op_idx](cand, k, _RNG)
        repair(cand, removed, _RNG, deadline)
        # 완전성 가드: 모든 블록이 배치된 해만 유효(미배정 해는 obj가 낮게 나와 오채택될 수 있음)
        if len(cand.placed) != n:
            continue
        new_obj = C.objective(cand)[0]

        threshold = best_obj * (1.0 + dev)
        reward = 0.0
        if new_obj <= threshold + 1e-9:
            current = cand; cur_obj = new_obj; accepts += 1
            if new_obj < best_obj - 1e-9:
                best = C.clone_state(cand); best_obj = new_obj
                improves += 1
                reward = 1.0
            else:
                reward = 0.3
        # 적응형 가중치 갱신
        weights[op_idx] = decay * weights[op_idx] + (1 - decay) * (1.0 + 4 * reward)

    if verbose:
        o = C.objective(best)
        print(f"[lns] iters={iters} accepts={accepts} improves={improves} "
              f"final obj={o[0]:.0f} (Z1={o[1]:.0f} Z2={o[2]:.0f} Z3={o[3]:.0f}) "
              f"in {time.perf_counter()-t0:.1f}s")
    return C.build_operations(best)


# rng 헬퍼 (모듈 전역 RNG — 재현성) ----------------------------------------
_RNG = random.Random(12345)


def roulette(weights: list) -> int:
    total = sum(weights)
    r = _RNG.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def rng_k(weights, kmax) -> int:
    return _RNG.randint(2, kmax)


# 공개 진입점 ----------------------------------------------------------------
def solve(prob_info: dict, timelimit: float = 60.0) -> dict:
    global _RNG
    _RNG = random.Random(12345)
    P.clear_cache()          # 인스턴스별 기하 캐시 격리(여러 인스턴스 연속 실행 안전)
    return lns(prob_info, timelimit=timelimit, verbose=False)


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "example_B2_b10.json"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    inst = json.load(open(path, encoding="utf-8"))
    _RNG = random.Random(12345)
    base = C.objective(C.build_state(inst, None))[0]
    print(f"[compare] 구성기 단독 obj={base:.0f}")
    lns(inst, timelimit=tl, verbose=True)
