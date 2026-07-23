# myalgorithm.py
# OGC2026 -- 자체 구성 휴리스틱 + LNS(Large Neighborhood Search)
#
# 구성:
#   Phase A: EDD 순 best-insertion 으로 feasible 초기해 직접 구성
#            (삽입 시 양방향 크레인 체크 -> 사후 repair 불필요, 시간 가드 내장)
#   Phase B: 남은 시간 동안 destroy & repair (LNS)
#            - destroy: tardy 중심 / bay-시간 클러스터 / 랜덤
#            - repair : 고속 best-insertion (시간후보 바깥 루프 + AABB 프리필터)
#   안전장치: 최종 check_feasibility 통과 못 하면 baseline greedy 로 폴백
#
# 제출 zip: myalgorithm.py + baseline_greedy.py + utils.py 모두 포함할 것.

import math
import random
import time

import baseline_greedy as bg
from utils import (
    Bay, Block,
    check_entry, check_exit, check_collisions, check_feasibility,
)

# =============================================================================
# utils 런타임 성능 패치
#   README 규정("utils.py 를 수정하지 말 것")에 따라 utils.py 파일은 원본 그대로
#   두고, 여기서 캐싱만 얹는다. 판정 결과는 원본과 완전히 동일하며(순수 캐싱),
#   check_feasibility 를 포함한 모든 판정이 원본 의미론을 유지한다.
# =============================================================================
import utils as _u


def _cached_bounding_rect(self):
    br = getattr(self, "_brect", None)
    if br is not None:
        return br
    layers = self.layers_at_pos()
    if not layers:
        br = (float(self.x), float(self.y),
              float(self.x) + 1.0, float(self.y) + 1.0)
    else:
        all_verts = [v for layer in layers for v in layer]
        br = _u._bounding_box(all_verts)
    object.__setattr__(self, "_brect", br)
    return br


def _blk_polys_at_pos(self):
    ps = getattr(self, "_polys", None)
    if ps is None:
        ps = [_u._poly_from_verts(l) for l in self._layers_cache]
        object.__setattr__(self, "_polys", ps)
    return ps


def _fast_check_entry(bay, blocks, new_block, fast=False):
    """utils.check_entry 와 동일 판정 + 블록별 폴리곤/박스 캐시 사용."""
    results = []
    if not bay.contains_block(new_block):
        bb = new_block.bounding_rect()
        bay_poly = _u._poly_from_verts(
            [[0, 0], [bay.width, 0], [bay.width, bay.height], [0, bay.height]])
        new_poly = _u._poly_from_verts(
            [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]])
        if bay_poly is not None and new_poly is not None:
            outside = new_poly.difference(bay_poly)
            if not outside.is_empty and outside.area > 0:
                results.append(_u.EntryObstruction(
                    existing_block=new_block, new_layer=0, exist_layer=0,
                    intersection=outside))
        return results

    new_bbox  = new_block.bounding_rect()
    new_polys = new_block.polys_at_pos()
    n_new     = len(new_polys)

    for exist in blocks:
        if not _u._bb_overlap(new_bbox, exist.bounding_rect()):
            continue
        exist_polys = exist.polys_at_pos()
        n_exist     = len(exist_polys)
        for k in range(n_new):
            poly_new = new_polys[k]
            if poly_new is None:
                continue
            for j in range(k, n_exist):
                poly_exist = exist_polys[j]
                if poly_exist is None:
                    continue
                try:
                    inter = poly_new.intersection(poly_exist)
                except Exception:
                    continue
                if not inter.is_empty and inter.area > 0:
                    obs = _u.EntryObstruction(
                        existing_block=exist, new_layer=k, exist_layer=j,
                        intersection=inter)
                    if fast:
                        return [obs]
                    results.append(obs)
    return results


def _fast_check_exit(bay, blocks, target_block, fast=False):
    """utils.check_exit 와 동일 판정 + 캐시 사용."""
    results = []
    target_bbox  = target_block.bounding_rect()
    target_polys = target_block.polys_at_pos()
    n_target     = len(target_polys)

    for exist in blocks:
        if exist.block_id == target_block.block_id:
            continue
        if not _u._bb_overlap(target_bbox, exist.bounding_rect()):
            continue
        exist_polys = exist.polys_at_pos()
        n_exist     = len(exist_polys)
        for k in range(n_target):
            poly_target = target_polys[k]
            if poly_target is None:
                continue
            for j in range(k, n_exist):
                poly_exist = exist_polys[j]
                if poly_exist is None:
                    continue
                try:
                    inter = poly_target.intersection(poly_exist)
                except Exception:
                    continue
                if not inter.is_empty and inter.area > 0:
                    obs = _u.EntryObstruction(
                        existing_block=exist, new_layer=k, exist_layer=j,
                        intersection=inter)
                    if fast:
                        return [obs]
                    results.append(obs)
    return results


def _fast_check_collisions(bay, blocks, layer_indices=None):
    """utils.check_collisions 와 동일 판정 + 캐시 사용."""
    results = []
    n = len(blocks)
    bboxes    = [b.bounding_rect() for b in blocks]
    all_polys = [b.polys_at_pos() for b in blocks]
    for i in range(n):
        for j in range(i + 1, n):
            if not _u._bb_overlap(bboxes[i], bboxes[j]):
                continue
            polys_a, polys_b = all_polys[i], all_polys[j]
            for k in range(min(len(polys_a), len(polys_b))):
                if layer_indices is not None and k not in layer_indices:
                    continue
                poly_a, poly_b = polys_a[k], polys_b[k]
                if poly_a is None or poly_b is None:
                    continue
                try:
                    inter = poly_a.intersection(poly_b)
                except Exception:
                    continue
                if not inter.is_empty and inter.area > 0:
                    results.append(_u.CollisionResult(
                        block_a=blocks[i], block_b=blocks[j],
                        layer_index=k, intersection=inter))
    return results


# 패치 장착: utils / baseline_greedy / 이 모듈의 참조를 모두 교체
_u.Block.bounding_rect = _cached_bounding_rect
_u.Block.polys_at_pos  = _blk_polys_at_pos
_u.check_entry      = _fast_check_entry
_u.check_exit       = _fast_check_exit
_u.check_collisions = _fast_check_collisions
bg.check_entry      = _fast_check_entry
bg.check_exit       = _fast_check_exit
bg.check_collisions = _fast_check_collisions
check_entry      = _fast_check_entry
check_exit       = _fast_check_exit
check_collisions = _fast_check_collisions



# 탐색 상한 (속도-품질 트레이드오프)
_MAX_POS   = 64    # (bay, orient, time) 당 위치 후보 상한
_MAX_TIMES = 60    # (bay) 당 시간 후보 상한




def _assignments_from_ops(operations: dict) -> dict[int, dict]:
    """operations dict -> {block_id: assignment dict} 역변환."""
    asg = {}
    for t_str, ops in operations.items():
        t = int(t_str)
        for op in ops:
            bid = op["block_id"]
            a = asg.setdefault(bid, {"block_id": bid})
            if op["type"] == "ENTRY":
                a["bay_id"] = op["bay_id"]
                a["x"] = int(op["x"]); a["y"] = int(op["y"])
                a["orient_idx"] = op["orient_idx"]; a["entry_time"] = t
            else:
                a["exit_time"] = t
    return asg


# =============================================================================
# 목적함수
# =============================================================================

def _obj2_of_loads(loads, u):
    if len(loads) < 2:
        return 0.0
    v = [u[j] * loads[j] for j in range(len(loads))]
    return float(math.floor(max(v) - min(v)))


def _objective(asg, blocks_data, u, w1, w2, w3, n_bays):
    obj1 = obj3 = 0.0
    loads = [0.0] * n_bays
    for a in asg.values():
        blk = blocks_data[a["block_id"]]
        obj1 += max(0.0, a["exit_time"] - blk["due_date"])
        loads[a["bay_id"]] += blk["workload"]
        obj3 += max(blk["bay_preferences"]) - blk["bay_preferences"][a["bay_id"]]
    obj2 = _obj2_of_loads(loads, u)
    return w1 * obj1 + w2 * obj2 + w3 * obj3, obj1, obj2, obj3


def _rects_overlap(r1, r2):
    return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])


# =============================================================================
# 고속 best-insertion
#
# 루프 구조: bay -> orient -> "시간 후보(오름차순)" -> 남은 위치 후보
#   * 각 위치는 가장 이른 feasible 시각만 취함 (성공 시 이후 시각에서 제외)
#   * AABB 프리필터로 시간중첩 + 공간중첩인 블록에 대해서만 shapely 체크
#   * 양방향 체크: 기존 블록이 나를 막는지 + 내가 기존 블록의 entry/exit 를 막는지
#   * 경계 시각(t 동일 이벤트)은 보수적으로 '동시 존재' 취급 -> feasible 보장 강화
# =============================================================================

def _best_insertion(bi, blocks_data, bays, bay_blocks, bay_sched,
                    loads, u, w1, w2, w3, best_score=float("inf"),
                    banned=None):
    """
    블록 bi 의 최적 삽입 탐색.

    w1(지각 가중치)이 지배적이라는 사실을 이용:
      bay/orient 별로 시간 후보를 오름차순 스캔하다가 feasible 위치가 하나라도
      나오는 "가장 이른 시각"을 찾으면 그 시각의 최적 위치로 확정하고 시간
      루프를 종료한다.  위치 후보는 해당 시간창과 겹치는 블록들로부터만 생성.
    """
    blk_data = blocks_data[bi]
    r_time = blk_data["release_time"]
    due    = blk_data["due_date"]
    proc   = blk_data["processing_time"]
    wload  = blk_data["workload"]
    prefs  = blk_data["bay_preferences"]
    s_max  = max(prefs)
    n_bays = len(bays)

    best = None

    for bay_id in sorted(range(n_bays), key=lambda j: prefs[j], reverse=True):
        bay     = bays[bay_id]
        sched_j = bay_sched[bay_id]
        blks_j  = bay_blocks[bay_id]

        loads2 = loads[:]
        loads2[bay_id] += wload
        obj2_new = _obj2_of_loads(loads2, u)
        base = w2 * obj2_new + w3 * (s_max - prefs[bay_id])

        # bay 하한 가지치기
        if w1 * max(0.0, r_time + proc - due) + base >= best_score:
            continue

        # bay 내 블록 캐시: (bid, a, e, Block, bounding_rect)
        entries = [(b, a, e, blks_j[b], blks_j[b].bounding_rect())
                   for b, (a, e) in sched_j.items()]

        time_cands = sorted({r_time} | {e for (_, _, e, _, _) in entries if e > r_time})
        time_cands = time_cands[:_MAX_TIMES]

        # orient 별 bbox 사전계산 + bay 에 들어가는지 확인
        orient_bbs = []
        for oi in range(len(blk_data["shape"])):
            bb = bg._block_bbox(blk_data, oi)
            lx0, ly0, lx1, ly1 = bb
            if (math.ceil(-lx0) > math.floor(bay.width - lx1) or
                    math.ceil(-ly0) > math.floor(bay.height - ly1)):
                continue
            orient_bbs.append((oi, bb))
        if not orient_bbs:
            continue

        for ec in time_cands:
            entry  = max(r_time, ec)
            exit_t = entry + proc
            tard   = max(0.0, exit_t - due)
            sc_t   = w1 * tard + base
            if sc_t >= best_score:
                break  # 시간 오름차순 -> 이후는 전부 열등

            # 이 시간창과 겹치는 블록 + 관계 태그
            ov = []
            window_blks = []
            for (oid, a, e, blk_o, rect_o) in entries:
                if not (a < exit_t and entry < e):
                    continue
                # 체커 의미론: 동시각 이벤트는 EXIT 먼저, ENTRY 는 block_id 오름차순
                at_entry = (a < entry < e) or (a == entry and oid < bi)
                at_exit  = (a < exit_t < e) or (e == exit_t and oid > bi)
                ent_in   = (entry < a < exit_t) or (a == entry and bi < oid)
                exi_in   = (entry < e < exit_t) or (e == exit_t and bi > oid)
                ov.append((blk_o, rect_o, at_entry, at_exit, ent_in, exi_in))
                window_blks.append(blk_o)

            found_here = None  # (top_y, cx, cy, oi, new_blk)
            for oi, bb in orient_bbs:
                lx0, ly0, lx1, ly1 = bb
                positions = bg._candidate_positions(
                    bay.width, bay.height, window_blks, bb)
                if len(positions) > _MAX_POS:
                    positions = positions[:_MAX_POS]

                for (cx, cy) in positions:
                    top_y = cy + ly1
                    if found_here is not None and top_y >= found_here[0]:
                        continue  # 이미 더 낮은 top_y 확보
                    new_rect = (cx + lx0, cy + ly0, cx + lx1, cy + ly1)
                    if not (new_rect[0] >= -1e-9 and new_rect[1] >= -1e-9 and
                            new_rect[2] <= bay.width + 1e-9 and
                            new_rect[3] <= bay.height + 1e-9):
                        continue
                    conflict = [o for o in ov if _rects_overlap(new_rect, o[1])]
                    new_blk = Block(block_id=bi, block_data=blk_data,
                                    x=cx, y=cy, orient_idx=oi)
                    if conflict:
                        ok = True
                        p_entry = [o[0] for o in conflict if o[2]]
                        if p_entry and check_entry(bay, p_entry, new_blk, fast=True):
                            ok = False
                        if ok:
                            p_exit = [new_blk] + [o[0] for o in conflict if o[3]]
                            if len(p_exit) > 1 and check_exit(bay, p_exit, new_blk, fast=True):
                                ok = False
                        if ok:
                            for (blk_o, _, _, _, ent_in, exi_in) in conflict:
                                if ent_in and check_entry(bay, [new_blk], blk_o, fast=True):
                                    ok = False; break
                                if exi_in and check_exit(bay, [new_blk, blk_o], blk_o, fast=True):
                                    ok = False; break
                                if check_collisions(bay, [new_blk, blk_o]):
                                    ok = False; break
                        if not ok:
                            continue
                    if found_here is None or top_y < found_here[0]:
                        found_here = (top_y, cx, cy, oi, new_blk)

            if found_here is not None:
                if banned is not None and banned == (bay_id, entry):
                    continue  # 직전 슬롯 금지 -> 다음 시각 탐색 (다양화)
                top_y, cx, cy, oi, new_blk = found_here
                score = sc_t + 1e-4 * top_y
                if score < best_score:
                    best_score = score
                    best = (bay_id, cx, cy, oi, entry, exit_t, new_blk)
                break  # 이 bay 의 가장 이른 feasible 시각 확정

    return best, best_score


# =============================================================================
# 강제 배치 (빈 bay 시간창 -- 항상 feasible)
# =============================================================================

def _force_insert(bi, blocks_data, bays, bay_sched):
    blk_data = blocks_data[bi]
    r_time = blk_data["release_time"]
    proc   = blk_data["processing_time"]
    prefs  = blk_data["bay_preferences"]

    for bay_id in sorted(range(len(bays)), key=lambda j: prefs[j], reverse=True):
        bay = bays[bay_id]
        for oi in range(len(blk_data["shape"])):
            lx0, ly0, lx1, ly1 = bg._block_bbox(blk_data, oi)
            px_lo, px_hi = math.ceil(-lx0), math.floor(bay.width - lx1)
            py_lo, py_hi = math.ceil(-ly0), math.floor(bay.height - ly1)
            if px_lo > px_hi or py_lo > py_hi:
                continue
            px, py = max(0, px_lo), max(0, py_lo)
            sched = list(bay_sched[bay_id].values())
            entry = bg._empty_bay_entry(sched, r_time, proc)
            new_blk = Block(block_id=bi, block_data=blk_data,
                            x=px, y=py, orient_idx=oi)
            return (bay_id, px, py, oi, entry, entry + proc, new_blk)
    raise RuntimeError(f"block {bi}: no valid position in any bay")


# =============================================================================
# 상태 조작 헬퍼
# =============================================================================

def _commit(bid, placement, cur, bay_blocks, bay_sched, loads, blocks_data):
    bay_id, cx, cy, oi, entry, exit_t, new_blk = placement
    cur[bid] = {"block_id": bid, "bay_id": bay_id,
                "x": int(cx), "y": int(cy), "orient_idx": oi,
                "entry_time": int(entry), "exit_time": int(exit_t)}
    bay_blocks[bay_id][bid] = new_blk
    bay_sched[bay_id][bid] = (int(entry), int(exit_t))
    loads[bay_id] += blocks_data[bid]["workload"]


def _remove(bid, cur, bay_blocks, bay_sched, loads, blocks_data):
    a = cur.pop(bid)
    j = a["bay_id"]
    blk = bay_blocks[j].pop(bid)
    del bay_sched[j][bid]
    loads[j] -= blocks_data[bid]["workload"]
    return a, blk


# =============================================================================
# Phase A: 초기해 구성 (EDD + best-insertion, feasible by construction)
# =============================================================================

def _construct(prob_info, bays, u, w1, w2, w3, t_start, t_guard):
    blocks_data = prob_info["blocks"]
    n_bays = len(bays)
    order = sorted(range(len(blocks_data)),
                   key=lambda i: (blocks_data[i]["due_date"],
                                  blocks_data[i]["processing_time"]))

    cur: dict[int, dict] = {}
    bay_blocks = [{} for _ in range(n_bays)]
    bay_sched  = [{} for _ in range(n_bays)]
    loads      = [0.0] * n_bays
    n_forced = 0

    for rank, bi in enumerate(order):
        if time.time() - t_start > t_guard:
            placement = _force_insert(bi, blocks_data, bays, bay_sched)
            n_forced += 1
        else:
            placement, _ = _best_insertion(bi, blocks_data, bays, bay_blocks,
                                           bay_sched, loads, u, w1, w2, w3)
            if placement is None:
                placement = _force_insert(bi, blocks_data, bays, bay_sched)
                n_forced += 1
        _commit(bi, placement, cur, bay_blocks, bay_sched, loads, blocks_data)

    return cur, bay_blocks, bay_sched, loads, n_forced


# =============================================================================
# Destroy 연산자
# =============================================================================

def _pick_destroy(asg, blocks_data, k, rng):
    ids = list(asg.keys())
    if k >= max(10, len(ids) // 4):
        # 대형 destroy: tardy 가중 랜덤 (부분 재시작)
        weights = [1.0 + max(0.0, asg[b]["exit_time"]
                             - blocks_data[b]["due_date"]) for b in ids]
        chosen = set()
        while len(chosen) < min(k, len(ids)):
            chosen.add(rng.choices(ids, weights=weights)[0])
        return list(chosen)
    mode = rng.random()

    def _neighbors(seed_id):
        sa = asg[seed_id]
        out = []
        for b, a in asg.items():
            if b == seed_id or a["bay_id"] != sa["bay_id"]:
                continue
            ov = (min(a["exit_time"], sa["exit_time"])
                  - max(a["entry_time"], sa["entry_time"]))
            if ov > 0:
                out.append((ov, b))
        out.sort(reverse=True)
        return [b for _, b in out]

    if mode < 0.45:
        tardy = [(max(0.0, a["exit_time"] - blocks_data[b]["due_date"]), b)
                 for b, a in asg.items()]
        tardy = [(t, b) for t, b in tardy if t > 0]
        if tardy:
            seed_id = rng.choices([b for _, b in tardy],
                                  weights=[t for t, _ in tardy])[0]
        else:
            seed_id = rng.choice(ids)
        return ([seed_id] + _neighbors(seed_id))[:k]
    elif mode < 0.75:
        seed_id = rng.choice(ids)
        return ([seed_id] + _neighbors(seed_id))[:k]
    else:
        return rng.sample(ids, min(k, len(ids)))


# =============================================================================
# Phase B: 개선 루프 (polish + LNS)
# =============================================================================

def _marginal_score(a, blocks_data, loads, u, w1, w2, w3):
    """블록의 현재 배치가 갖는 삽입점수(스케일 동일) -- 개선 판정 기준선."""
    bd = blocks_data[a["block_id"]]
    tard = max(0.0, a["exit_time"] - bd["due_date"])
    pen  = max(bd["bay_preferences"]) - bd["bay_preferences"][a["bay_id"]]
    return w1 * tard + w2 * _obj2_of_loads(loads, u) + w3 * pen


def _polish_pass(cur, bay_blocks, bay_sched, loads, bays, u,
                 blocks_data, w1, w2, w3, t_end, rng):
    """
    단일 블록 재배치 개선 스윕: 손해 볼 수 없는 확정 개선만 수행.
    블록을 빼고, 현재보다 '엄격히 좋은' 자리가 있으면 이동, 없으면 원위치.
    잠재 이득(w1*지각 + w3*선호패널티)이 큰 블록부터 처리.
    """
    n_moved = 0
    cand = []
    for bid, a in cur.items():
        bd = blocks_data[bid]
        tard = max(0.0, a["exit_time"] - bd["due_date"])
        pen  = max(bd["bay_preferences"]) - bd["bay_preferences"][a["bay_id"]]
        gain = w1 * tard + w3 * pen
        if gain > 0:
            cand.append((gain, bid))
    cand.sort(reverse=True)

    for _, bid in cand:
        if time.time() > t_end:
            break
        a, blk = _remove(bid, cur, bay_blocks, bay_sched, loads, blocks_data)
        bound = _marginal_score(a, blocks_data, loads, u, w1, w2, w3) - 1e-6
        placement, _ = _best_insertion(bid, blocks_data, bays, bay_blocks,
                                       bay_sched, loads, u, w1, w2, w3,
                                       best_score=bound)
        if placement is None:
            # 원위치 복구
            j = a["bay_id"]
            cur[bid] = a
            bay_blocks[j][bid] = blk
            bay_sched[j][bid] = (a["entry_time"], a["exit_time"])
            loads[j] += blocks_data[bid]["workload"]
        else:
            _commit(bid, placement, cur, bay_blocks, bay_sched, loads, blocks_data)
            n_moved += 1
    return n_moved


def _lns(prob_info, cur, bay_blocks, bay_sched, loads, bays, u,
         t_start, t_end, verbose=True):
    blocks_data = prob_info["blocks"]
    n_bays = len(bays)
    w = prob_info.get("weights", {})
    w1, w2, w3 = w.get("w1", 1.0), w.get("w2", 1.0), w.get("w3", 1.0)

    rng = random.Random(20260722)

    def _rebuild(asg):
        bb = [{} for _ in range(n_bays)]
        bs = [{} for _ in range(n_bays)]
        ld = [0.0] * n_bays
        for bid, a in asg.items():
            j = a["bay_id"]
            bb[j][bid] = Block(block_id=bid, block_data=blocks_data[bid],
                               x=int(a["x"]), y=int(a["y"]),
                               orient_idx=a["orient_idx"])
            bs[j][bid] = (a["entry_time"], a["exit_time"])
            ld[j] += blocks_data[bid]["workload"]
        return bb, bs, ld

    cur_obj, o1, o2, o3 = _objective(cur, blocks_data, u, w1, w2, w3, n_bays)
    if verbose:
        print(f"[LNS] start obj={cur_obj:.0f} (obj1={o1:.0f} obj2={o2:.0f} obj3={o3:.0f})"
              f"  budget={t_end - time.time():.1f}s", flush=True)

    # -- 1) polish 스윕: 확정 개선 -------------------------------------------
    n_moved = _polish_pass(cur, bay_blocks, bay_sched, loads, bays, u,
                           blocks_data, w1, w2, w3, t_end, rng)
    cur_obj, o1, o2, o3 = _objective(cur, blocks_data, u, w1, w2, w3, n_bays)
    if verbose:
        print(f"[LNS] polish moved={n_moved} -> obj={cur_obj:.0f}"
              f" (obj1={o1:.0f} obj2={o2:.0f} obj3={o3:.0f})"
              f"  elapsed={time.time() - t_start:.1f}s", flush=True)

    best = {b: dict(a) for b, a in cur.items()}
    best_obj = cur_obj

    # -- 2) destroy & repair -------------------------------------------------
    k_base = max(2, min(6, len(cur) // 40 + 2))
    it = accepted = no_improve = 0
    last_log = time.time()
    since_polish = 0
    ins_time_sum, ins_count = 0.0, 0   # 삽입 1회당 평균 시간 (대형 destroy 게이트)

    while time.time() < t_end:
        it += 1
        k = k_base + (2 if no_improve > 40 else 0)
        # 시간 인지형 대형 destroy: 여유가 있을 때만 부분 재시작 규모로 제거
        if rng.random() < 0.10 and ins_count >= 20:
            avg_ins = ins_time_sum / ins_count
            k_big = min(len(cur), max(10, len(cur) // 3))
            if k_big * avg_ins < min(5.0, 0.25 * (t_end - time.time())):
                k = k_big
        removed_ids = _pick_destroy(cur, blocks_data, k, rng)

        backup = {bid: _remove(bid, cur, bay_blocks, bay_sched, loads, blocks_data)
                  for bid in removed_ids}

        # 재삽입 순서: EDD / 지각 내림차순 / 셔플 혼합
        m = rng.random()
        if m < 0.4:
            order = sorted(removed_ids,
                           key=lambda b: (blocks_data[b]["due_date"],
                                          blocks_data[b]["processing_time"]))
        elif m < 0.75:
            order = sorted(removed_ids,
                           key=lambda b: -(max(0.0, backup[b][0]["exit_time"]
                                               - blocks_data[b]["due_date"])))
        else:
            order = list(removed_ids)
            rng.shuffle(order)

        use_ban = rng.random() < 0.5
        inserted, ok = [], True
        for bid in order:
            if time.time() > t_end:
                ok = False
                break
            ban = None
            if use_ban:
                pa = backup[bid][0]
                ban = (pa["bay_id"], pa["entry_time"])
            t_ins0 = time.time()
            placement, _ = _best_insertion(bid, blocks_data, bays, bay_blocks,
                                           bay_sched, loads, u, w1, w2, w3,
                                           banned=ban)
            ins_time_sum += time.time() - t_ins0
            ins_count += 1
            if placement is None:
                ok = False
                break
            _commit(bid, placement, cur, bay_blocks, bay_sched, loads, blocks_data)
            inserted.append(bid)

        new_obj = None
        if ok:
            new_obj, *_ = _objective(cur, blocks_data, u, w1, w2, w3, n_bays)

        if not ok or new_obj > cur_obj + 1e-9:
            for bid in inserted:
                _remove(bid, cur, bay_blocks, bay_sched, loads, blocks_data)
            for bid, (a, blk) in backup.items():
                j = a["bay_id"]
                cur[bid] = a
                bay_blocks[j][bid] = blk
                bay_sched[j][bid] = (a["entry_time"], a["exit_time"])
                loads[j] += blocks_data[bid]["workload"]
            no_improve += 1
        else:
            accepted += 1
            if new_obj < cur_obj - 1e-9:
                since_polish += 1
            cur_obj = new_obj
            if cur_obj < best_obj - 1e-9:
                best = {b: dict(a) for b, a in cur.items()}
                best_obj = cur_obj
                no_improve = 0
            else:
                no_improve += 1

        # D&R 이 뭔가 바꿨으면 주기적으로 polish 재실행
        if since_polish >= 8:
            since_polish = 0
            _polish_pass(cur, bay_blocks, bay_sched, loads, bays, u,
                         blocks_data, w1, w2, w3, t_end, rng)
            cur_obj, *_ = _objective(cur, blocks_data, u, w1, w2, w3, n_bays)
            if cur_obj < best_obj - 1e-9:
                best = {b: dict(a) for b, a in cur.items()}
                best_obj = cur_obj

        if no_improve > 150:
            cur = {b: dict(a) for b, a in best.items()}
            bay_blocks, bay_sched, loads = _rebuild(cur)
            cur_obj = best_obj
            no_improve = 0

        if verbose and time.time() - last_log > 5.0:
            last_log = time.time()
            print(f"[LNS] it={it} acc={accepted} cur={cur_obj:.0f} best={best_obj:.0f}"
                  f"  elapsed={time.time() - t_start:.1f}s", flush=True)

    if verbose:
        _, b1, b2, b3 = _objective(best, blocks_data, u, w1, w2, w3, n_bays)
        print(f"[LNS] done  it={it} acc={accepted}  best={best_obj:.0f}"
              f" (obj1={b1:.0f} obj2={b2:.0f} obj3={b3:.0f})", flush=True)
    return best



# =============================================================================
# baseline Phase-1 가속: 정확 가지치기(exact pruning) 버전 _place_blocks
#   - 결과는 baseline 과 완전히 동일 (하한이 정확하므로 잘리는 후보는 이길 수 없음)
#   - bg._place_blocks 에 몽키패치되어 greedyalgorithm 의 Phase 1 / repair 모두 가속
# =============================================================================

def _slot_earliest_pruned(new_blk, bay, placed_in_bay, schedule_in_bay,
                          r_time, proc, w1, due, score_base, best_score):
    """bg._find_earliest_slot 과 동일 + 지각 하한 컷오프."""
    candidate_entries = sorted({r_time} | {e for _, e in schedule_in_bay if e > r_time})

    for entry_candidate in candidate_entries:
        entry = max(r_time, entry_candidate)
        # 정확 하한: 이 시각 이후로는 어떤 위치도 best 를 이길 수 없음
        if w1 * max(0.0, entry + proc - due) + score_base >= best_score:
            return None, None
        exit_t = entry + proc

        present_at_entry = [
            b for b, (a, e) in zip(placed_in_bay, schedule_in_bay)
            if a < entry < e
        ]
        if check_entry(bay, present_at_entry, new_blk, fast=True):
            continue

        present_at_exit = [new_blk] + [
            b for b, (a, e) in zip(placed_in_bay, schedule_in_bay)
            if a < exit_t < e
        ]
        if check_exit(bay, present_at_exit, new_blk, fast=True):
            continue

        s4_blocked = False
        for b_other, (a_other, e_other) in zip(placed_in_bay, schedule_in_bay):
            if a_other < entry or e_other > exit_t:
                continue
            if not (entry < e_other and a_other < exit_t):
                continue
            if check_collisions(bay, [new_blk, b_other]):
                s4_blocked = True
                break
        if s4_blocked:
            continue

        return entry, exit_t

    return None, None


class _ConstructionTimeout(Exception):
    pass


def _fast_place_blocks(block_ids, blocks_data, bays,
                       bay_placed, bay_schedule, bay_loads,
                       w1, w2, w3, forced_ids,
                       prev_assignments=None, t_start=None, log_interval=0,
                       deadline=None):
    """bg._place_blocks 와 동일 시그니처/결과, 정확 가지치기로 가속.
    deadline 초과 시 _ConstructionTimeout (대안 정렬 구성 전용)."""
    n_bays  = len(bays)
    n_total = len(block_ids)
    result  = {}
    n_forced = n_fallback = 0

    _bay_areas  = [bay.width * bay.height for bay in bays]
    _avg_area   = sum(_bay_areas) / n_bays
    bay_weights = [_avg_area / a for a in _bay_areas]

    for rank, bi in enumerate(block_ids):
        if deadline is not None and time.time() > deadline:
            raise _ConstructionTimeout()
        blk_data = blocks_data[bi]
        r_time   = blk_data["release_time"]
        due      = blk_data["due_date"]
        proc     = blk_data["processing_time"]
        workload = blk_data["workload"]
        prefs    = blk_data["bay_preferences"]
        s_max    = max(prefs)
        n_orient = len(blk_data["shape"])

        best_score     = float("inf")
        best_placement = None
        used_forced    = bi in forced_ids

        if not used_forced:
            # -- repair fast-path (baseline 과 동일) --------------------------
            if prev_assignments and bi in prev_assignments:
                pa = prev_assignments[bi]
                pb_id = pa["bay_id"]
                px, py, poi = int(pa["x"]), int(pa["y"]), pa["orient_idx"]
                prev_blk = Block(block_id=bi, block_data=blk_data,
                                 x=px, y=py, orient_idx=poi)
                if bays[pb_id].contains_block(prev_blk):
                    entry, exit_t = bg._find_earliest_slot(
                        prev_blk, bays[pb_id],
                        bay_placed[pb_id], bay_schedule[pb_id],
                        r_time, proc)
                    if entry is not None:
                        tardiness = max(0.0, exit_t - due)
                        p_bb = bg._block_bbox(blk_data, poi)
                        best_score = bg._placement_score(
                            tardiness, workload, bay_loads, pb_id,
                            s_max - prefs[pb_id], bay_weights, w1, w2, w3,
                            top_y=py + p_bb[3])
                        best_placement = (pb_id, px, py, poi, entry, exit_t)

            # -- 전탐색 + 정확 가지치기 ---------------------------------------
            bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
            for bay_id in bay_order:
                bay             = bays[bay_id]
                placed_in_bay   = bay_placed[bay_id]
                schedule_in_bay = bay_schedule[bay_id]

                # bay 상수부 (obj2 근사 + 선호도) -- baseline 점수식과 동일
                new_load = bay_loads[bay_id] + workload
                obj2_new = max(
                    (abs(bay_weights[bay_id] * new_load
                         - bay_weights[j] * bay_loads[j])
                     for j in range(n_bays) if j != bay_id),
                    default=0.0)
                score_base = w2 * obj2_new + w3 * (s_max - prefs[bay_id])

                # bay 하한 가지치기
                if w1 * max(0.0, r_time + proc - due) + score_base >= best_score:
                    continue

                for oi in range(n_orient):
                    blk_bb = bg._block_bbox(blk_data, oi)
                    lx0_oi, ly0_oi, lx1_oi, ly1_oi = blk_bb
                    if (math.ceil(-lx0_oi) > math.floor(bay.width  - lx1_oi) or
                            math.ceil(-ly0_oi) > math.floor(bay.height - ly1_oi)):
                        continue

                    active_in_bay = [
                        b for b, (a_k, e_k) in zip(placed_in_bay, schedule_in_bay)
                        if e_k > r_time
                    ]
                    candidates = bg._candidate_positions(
                        bay.width, bay.height, active_in_bay, blk_bb)
                    for (cx, cy) in candidates:
                        new_blk = Block(block_id=bi, block_data=blk_data,
                                        x=cx, y=cy, orient_idx=oi)
                        if not bay.contains_block(new_blk):
                            continue

                        entry, exit_t = _slot_earliest_pruned(
                            new_blk, bay, placed_in_bay, schedule_in_bay,
                            r_time, proc, w1, due, score_base, best_score)
                        if entry is None:
                            continue

                        tardiness = max(0.0, exit_t - due)
                        score = bg._placement_score(
                            tardiness, workload, bay_loads, bay_id,
                            s_max - prefs[bay_id], bay_weights, w1, w2, w3,
                            top_y=cy + blk_bb[3])
                        if score < best_score:
                            best_score     = score
                            best_placement = (bay_id, cx, cy, oi, entry, exit_t)

        if best_placement is None:
            best_placement = bg._force_place(bi, blocks_data, bays, bay_schedule, prefs)
            n_fallback += 1
        if used_forced:
            n_forced += 1

        bay_id, cx, cy, oi, entry, exit_t = best_placement
        final_blk = Block(block_id=bi, block_data=blk_data, x=cx, y=cy, orient_idx=oi)
        bay_placed[bay_id].append(final_blk)
        bay_schedule[bay_id].append((entry, exit_t))
        bay_loads[bay_id] += workload

        result[bi] = {
            "block_id":   bi,
            "bay_id":     bay_id,
            "x":          int(round(cx)),
            "y":          int(round(cy)),
            "orient_idx": oi,
            "entry_time": int(round(entry)),
            "exit_time":  int(round(exit_t)),
        }

        if log_interval > 0 and t_start is not None:
            n_done = rank + 1
            if n_done % log_interval == 0 or n_done == n_total:
                elapsed = time.time() - t_start
                print(f"[Greedy]   {n_done:4d}/{n_total}"
                      f"  block{bi:<4d} -> bay{bay_id} ({cx},{cy}) oi={oi}"
                      f"  t=[{int(round(entry))},{int(round(exit_t))})"
                      f"  fallback={n_fallback}  {elapsed:.1f}s", flush=True)

    return result


# greedyalgorithm 의 Phase 1 / repair 가 가속 버전을 쓰도록 몽키패치
bg._place_blocks = _fast_place_blocks



# =============================================================================
# 다중 정렬 구성: EDD 외 대안 순서로 구성을 추가 시도, 최선을 시작점으로
# =============================================================================

def _greedy_with_order(prob_info, sorted_indices, budget, deadline=None):
    """지정된 블록 순서로 Phase1(가속) + Phase2(repair) 실행."""
    t_local = time.time()
    bays_data   = prob_info["bays"]
    blocks_data = prob_info["blocks"]
    n_bays      = len(bays_data)
    w = prob_info.get("weights", {})
    w1, w2, w3 = w.get("w1", 1.0), w.get("w2", 1.0), w.get("w3", 1.0)

    bays = [Bay.from_dict(d, i) for i, d in enumerate(bays_data)]
    bay_placed   = [[] for _ in range(n_bays)]
    bay_schedule = [[] for _ in range(n_bays)]
    bay_loads    = [0.0] * n_bays

    assignments = _fast_place_blocks(
        sorted_indices, blocks_data, bays,
        bay_placed, bay_schedule, bay_loads,
        w1, w2, w3, forced_ids=set(), deadline=deadline)

    sol = {"operations": bg._build_operations(list(assignments.values()))}
    assignments = bg._repair(prob_info, sol, assignments, bays, blocks_data,
                             w1, w2, w3, t_local, budget, repair_mode="greedy")
    return {"operations": bg._build_operations(list(assignments.values()))}


_ALT_ORDERS = [
    ("slack",    lambda bd: (bd["due_date"] - bd["release_time"] - bd["processing_time"],
                             bd["due_date"])),
    ("edd-wl",   lambda bd: (bd["due_date"], -bd["workload"])),
    ("rel-edd",  lambda bd: (bd["release_time"], bd["due_date"])),
]


# =============================================================================
# 엔트리 포인트
# =============================================================================

def algorithm(prob_info, timelimit=60):
    t_start = time.time()
    bays_data = prob_info["bays"]
    n_bays    = len(bays_data)
    w = prob_info.get("weights", {})
    w1, w2, w3 = w.get("w1", 1.0), w.get("w2", 1.0), w.get("w3", 1.0)

    bays  = [Bay.from_dict(bays_data[j], j) for j in range(n_bays)]
    areas = [b.width * b.height for b in bays]
    avg_a = sum(areas) / n_bays
    u     = [avg_a / a for a in areas]

    # -- Phase A: baseline greedy 구성 (검증된 품질) -------------------------
    sol_a = bg.greedyalgorithm(prob_info, timelimit=timelimit * 0.8)
    res_a = check_feasibility(prob_info, sol_a)
    if not res_a["feasible"]:
        print("[MyAlg] baseline solution infeasible -- returning as-is", flush=True)
        return sol_a

    print(f"[MyAlg] construct obj={res_a['objective']:.0f}"
          f" (obj1={res_a['obj1']:.0f} obj2={res_a['obj2']:.0f} obj3={res_a['obj3']:.0f})"
          f"  elapsed={time.time() - t_start:.1f}s", flush=True)

    fallback_sol, fallback_obj = sol_a, res_a["objective"]
    t_cons1 = time.time() - t_start

    # -- 대안 정렬 구성: 시간이 허락하는 만큼 추가 시도, 최선 유지 ------------
    blocks_data_ = prob_info["blocks"]
    for oname, okey in _ALT_ORDERS:
        elapsed_now = time.time() - t_start
        est = 1.3 * max(t_cons1, 0.5)
        # 대안 구성은 '저렴'할 때만: 예산의 55% 안에 끝날 전망이어야 함
        if elapsed_now + est > 0.55 * timelimit:
            break
        order = sorted(range(len(blocks_data_)),
                       key=lambda i: okey(blocks_data_[i]))
        alt_deadline = time.time() + 1.4 * max(t_cons1, 0.5) + 1.0
        try:
            sol_alt = _greedy_with_order(prob_info, order,
                                         budget=1.4 * max(t_cons1, 0.5) + 1.0,
                                         deadline=alt_deadline)
            res_alt = check_feasibility(prob_info, sol_alt)
        except _ConstructionTimeout:
            print(f"[MyAlg] alt-order {oname}: timeout -- discarded", flush=True)
            continue
        except Exception as e:
            print(f"[MyAlg] alt-order {oname} error: {e!r}", flush=True)
            continue
        if res_alt["feasible"]:
            tag = "better!" if res_alt["objective"] < fallback_obj else ""
            print(f"[MyAlg] alt-order {oname}: obj={res_alt['objective']:.0f} {tag}",
                  flush=True)
            if res_alt["objective"] < fallback_obj:
                fallback_sol, fallback_obj = sol_alt, res_alt["objective"]
                sol_a = sol_alt

    # -- LNS 예산 산정 (최종 검증 시간 확보) ---------------------------------
    tv0 = time.time()
    check_feasibility(prob_info, sol_a)
    t_check = time.time() - tv0
    reserve = max(1.0, 2.5 * t_check) + 0.03 * timelimit
    t_end = t_start + timelimit - reserve
    if time.time() >= t_end:
        return fallback_sol

    # -- Phase B: polish + LNS ----------------------------------------------
    cur = _assignments_from_ops(sol_a["operations"])
    blocks_data = prob_info["blocks"]
    bay_blocks = [{} for _ in range(n_bays)]
    bay_sched  = [{} for _ in range(n_bays)]
    loads      = [0.0] * n_bays
    for bid, a in cur.items():
        j = a["bay_id"]
        bay_blocks[j][bid] = Block(block_id=bid, block_data=blocks_data[bid],
                                   x=int(a["x"]), y=int(a["y"]),
                                   orient_idx=a["orient_idx"])
        bay_sched[j][bid] = (a["entry_time"], a["exit_time"])
        loads[j] += blocks_data[bid]["workload"]

    try:
        best_asg = _lns(prob_info, cur, bay_blocks, bay_sched, loads,
                        bays, u, t_start, t_end)
    except Exception as e:
        print(f"[MyAlg] LNS error: {e!r} -- fallback", flush=True)
        return fallback_sol

    best_sol = {"operations": bg._build_operations(list(best_asg.values()))}
    res_b = check_feasibility(prob_info, best_sol)
    if res_b["feasible"] and res_b["objective"] <= fallback_obj:
        print(f"[MyAlg] final obj={res_b['objective']:.0f}"
              f"  ({fallback_obj:.0f} -> {res_b['objective']:.0f},"
              f" {100 * (1 - res_b['objective'] / max(fallback_obj, 1e-9)):.1f}% down)"
              f"  total={time.time() - t_start:.1f}s", flush=True)
        return best_sol
    print(f"[MyAlg] LNS result rejected (feasible={res_b['feasible']}) -- fallback", flush=True)
    return fallback_sol


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("instance")
    p.add_argument("--timelimit", type=float, default=60.0)
    args = p.parse_args()
    with open(args.instance) as f:
        prob = json.load(f)
    sol = algorithm(prob, timelimit=args.timelimit)
    r = check_feasibility(prob, sol)
    print(f"FINAL feasible={r['feasible']} obj={r.get('objective', float('nan'))}")
