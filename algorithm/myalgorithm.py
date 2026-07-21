# myalgorithm.py
# ---------------------------------------------------------------------------
# Phase 1+2 : baseline EDD greedy + repair  (baseline_greedy.py, unchanged)
# Phase 3   : Large Neighborhood Search (ruin & recreate) on the remaining time
#
# LNS iteration:
#   ruin     -- remove a subset of blocks, chosen by one of three operators:
#                 * "worst"  : blocks with the largest objective contribution
#                              (w1*tardiness + w3*preference penalty) + random
#                 * "window" : all blocks of 1-2 bays whose time interval
#                              intersects a random time window (re-sequences a
#                              whole congested period)
#                 * "random" : uniform random subset
#   recreate -- re-insert the removed blocks with the baseline placement
#               kernel in a randomized-EDD order
#   accept   -- only if the cheap objective improves AND the official
#               check_feasibility verifies the candidate.
#
# The best *verified* solution is always returned.  Phase 3 is wrapped in a
# try/except: any unexpected error falls back to the Phase-1+2 solution, so
# the improvement phase can never turn a feasible run into a failure.
# ---------------------------------------------------------------------------

import math
import random
import time


def algorithm(prob_info, timelimit=60):
    """Entry point. Signature must not change."""
    t0 = time.time()

    import baseline_greedy as bg
    from utils import check_feasibility

    # ------------------------------------------------------------------
    # Phase 1+2: baseline greedy construction (~45% of the budget)
    # ------------------------------------------------------------------
    greedy_budget = max(5.0, timelimit * 0.45)
    sol = bg.greedyalgorithm(prob_info, greedy_budget)

    res = check_feasibility(prob_info, sol)
    if not res["feasible"]:
        return sol                      # nothing feasible to improve on

    try:
        return _lns_improve(prob_info, sol, res, t0, timelimit, bg,
                            check_feasibility)
    except Exception as e:              # never lose the feasible base
        print(f"[LNS] aborted by exception: {e!r} -- returning greedy solution")
        return sol


# ---------------------------------------------------------------------------
# Phase 3: LNS improvement
# ---------------------------------------------------------------------------

def _lns_improve(prob_info, sol, res, t0, timelimit, bg, check_feasibility):
    from utils import Bay, Block

    rng = random.Random(12345)

    blocks_data = prob_info["blocks"]
    bays_data   = prob_info["bays"]
    n_blocks    = len(blocks_data)
    n_bays      = len(bays_data)
    w = prob_info.get("weights", {})
    w1, w2, w3 = w.get("w1", 1.0), w.get("w2", 1.0), w.get("w3", 1.0)

    best_sol = sol
    best_obj = res["objective"]
    print(f"[LNS] start obj={best_obj:.0f} "
          f"(obj1={res['obj1']:.0f} obj2={res['obj2']:.0f} obj3={res['obj3']:.0f})")

    # -- rebuild assignments from operations ---------------------------------
    assignments = {}
    for t_str, ops in sol["operations"].items():
        t = int(t_str)
        for op in ops:
            bi = op["block_id"]
            a = assignments.setdefault(bi, {"block_id": bi})
            if op["type"] == "ENTRY":
                a.update(bay_id=op["bay_id"], x=op["x"], y=op["y"],
                         orient_idx=op["orient_idx"], entry_time=t)
            else:
                a["exit_time"] = t

    bays        = [Bay.from_dict(d, i) for i, d in enumerate(bays_data)]
    best_assign = {k: dict(v) for k, v in assignments.items()}

    bay_areas = [b.width * b.height for b in bays]
    avg_area  = sum(bay_areas) / n_bays
    u         = [avg_area / a for a in bay_areas]

    def contribution(bi, a):
        bd = blocks_data[bi]
        return (w1 * max(0.0, a["exit_time"] - bd["due_date"])
                + w3 * (max(bd["bay_preferences"])
                        - bd["bay_preferences"][a["bay_id"]]))

    def cheap_obj(asg):
        o1 = o3 = 0.0
        loads = [0.0] * n_bays
        for bi, a in asg.items():
            bd = blocks_data[bi]
            o1 += max(0.0, a["exit_time"] - bd["due_date"])
            o3 += max(bd["bay_preferences"]) - bd["bay_preferences"][a["bay_id"]]
            loads[a["bay_id"]] += bd["workload"]
        if n_bays >= 2:
            vals = [u[j] * loads[j] for j in range(n_bays)]
            o2 = math.floor(max(vals) - min(vals))
        else:
            o2 = 0.0
        return w1 * o1 + w2 * o2 + w3 * o3

    # -- LNS parameters -------------------------------------------------------
    deadline = t0 + timelimit * 0.92
    k_min    = 3
    k_max    = max(6, min(14, max(4, n_blocks // 8)))
    it = accepted = 0
    avg_iter = 0.0

    # Cap candidate positions per (bay, orient) during LNS so each recreate
    # call stays fast even in crowded bays.
    _orig_cand = bg._candidate_positions
    def _capped_cand(bw, bh, placed, bb, _cap=48):
        c = _orig_cand(bw, bh, placed, bb)
        if len(c) > _cap:
            c = sorted(c, key=lambda p: (p[1], p[0]))[:_cap]
        return c
    bg._candidate_positions = _capped_cand

    try:
        while time.time() + avg_iter < deadline:
            it_t0 = time.time()
            it += 1
            cur = {k: dict(v) for k, v in best_assign.items()}

            # ---------------- ruin ----------------
            op = rng.random()
            removal = set()
            if op < 0.60:
                # worst-contribution + random fill
                k = rng.randint(k_min, k_max)
                scored = sorted(((contribution(bi, a), bi)
                                 for bi, a in cur.items()), reverse=True)
                pool = [bi for c, bi in scored if c > 0][:max(2 * k, 10)]
                if pool:
                    removal.update(rng.sample(pool,
                                              min(max(1, k // 2), len(pool))))
                ids = list(cur.keys())
                while len(removal) < k:
                    removal.add(rng.choice(ids))
            elif op < 0.82:
                # time-window ruin: blocks of 1-2 bays hitting a random window
                t_hi = max(a["exit_time"] for a in cur.values())
                t_lo = min(a["entry_time"] for a in cur.values())
                span = max(4, (t_hi - t_lo) // 6)
                ws   = rng.uniform(t_lo, max(t_lo, t_hi - span))
                we   = ws + span
                nby  = 1 if n_bays == 1 else rng.randint(1, 2)
                tgt  = set(rng.sample(range(n_bays), min(nby, n_bays)))
                cand = [bi for bi, a in cur.items()
                        if a["bay_id"] in tgt
                        and a["entry_time"] < we and a["exit_time"] > ws]
                rng.shuffle(cand)
                removal.update(cand[:k_max])
                if not removal:
                    continue
            else:
                # pure random
                k = rng.randint(k_min, k_max)
                removal.update(rng.sample(list(cur.keys()),
                                          min(k, len(cur))))

            for bi in removal:
                cur.pop(bi, None)

            # ---------------- rebuild state ----------------
            bay_placed   = [[] for _ in range(n_bays)]
            bay_schedule = [[] for _ in range(n_bays)]
            bay_loads    = [0.0] * n_bays
            for a in cur.values():
                bi  = a["block_id"]
                blk = Block(block_id=bi, block_data=blocks_data[bi],
                            x=int(a["x"]), y=int(a["y"]),
                            orient_idx=a["orient_idx"])
                bay_placed[a["bay_id"]].append(blk)
                bay_schedule[a["bay_id"]].append((a["entry_time"],
                                                  a["exit_time"]))
                bay_loads[a["bay_id"]] += blocks_data[bi]["workload"]

            # ---------------- recreate ----------------
            noise = rng.uniform(0.0, 6.0)
            order = sorted(removal,
                           key=lambda b: (blocks_data[b]["due_date"]
                                          + rng.uniform(-noise, noise),
                                          blocks_data[b]["processing_time"]))
            partial = bg._place_blocks(order, blocks_data, bays,
                                       bay_placed, bay_schedule, bay_loads,
                                       w1, w2, w3, forced_ids=set())
            cur.update(partial)

            # ---------------- accept ----------------
            if cheap_obj(cur) < best_obj - 1e-9:
                cand_sol = {"operations":
                            bg._build_operations(list(cur.values()))}
                r = check_feasibility(prob_info, cand_sol)
                if r["feasible"] and r["objective"] < best_obj - 1e-9:
                    best_obj    = r["objective"]
                    best_sol    = cand_sol
                    best_assign = cur
                    accepted   += 1
                    print(f"[LNS] it={it} accept obj={best_obj:.0f} "
                          f"(obj1={r['obj1']:.0f} obj2={r['obj2']:.0f} "
                          f"obj3={r['obj3']:.0f}) t={time.time()-t0:.1f}s")

            dt = time.time() - it_t0
            avg_iter = dt if it == 1 else 0.7 * avg_iter + 0.3 * dt
    finally:
        bg._candidate_positions = _orig_cand

    print(f"[LNS] done  obj={best_obj:.0f}  iters={it}  accepted={accepted}  "
          f"t={time.time()-t0:.1f}s")
    return best_sol
