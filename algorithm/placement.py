"""
placement.py — 폴리곤 bottom-left-fill 배치 엔진 (2단계 모듈 A)
================================================================

역할
----
"베이 + 배치할 블록 하나 + 이미 놓인 장애물 목록"을 받아,
장애물과 footprint(평면 투영)가 겹치지 않는 정수 (x, y, 방향)을 찾아 돌려준다.
못 찾으면 None.

왜 footprint 비중첩인가
-----------------------
크레인 판정 규칙은 "새 블록 레이어 k가 기존 블록의 같거나 더 높은 레이어(j>=k)와
평면상 겹칠 때만 막힘"이다. 따라서 두 블록의 **전체 footprint(모든 레이어 합집합)**가
아예 안 겹치면:
  - 충돌 제약(같은 레벨 겹침) : 통과
  - 크레인 진입/퇴출 제약       : 통과
즉 비중첩 배치만 하면 공간·크레인 실현가능성이 *공짜로* 보장된다.
(서로 다른 레벨로 끼워 넣는 interlocking 은 더 빽빽하지만 위험 — 3단계에서 다룸.)

채점기와의 일관성
-----------------
utils.Block 으로 블록 기하를 만들고, 폴리곤은 utils 와 동일하게
invalid 일 때 buffer(0) 으로 보정한다. 좌표는 정수만 생성한다
(해의 x,y 는 정수여야 하고 채점기도 반올림하므로).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from utils import Block, Bay

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import translate as _shp_translate
from shapely.prepared import prep as _shp_prep

# 면적 겹침을 '충돌'로 볼 최소 임계값. 변(edge)만 닿는 건 collision-free 이므로
# intersection.area > EPS_AREA 일 때만 충돌로 본다.
EPS_AREA = 1e-9
_TOL = 1e-9


# ---------------------------------------------------------------------------
# 폴리곤 헬퍼 (채점기의 _poly_from_verts 동작을 미러링)
# ---------------------------------------------------------------------------
def _poly(verts) -> Optional[Polygon]:
    if not verts or len(verts) < 3:
        return None
    try:
        p = Polygon(verts)
        if not p.is_valid:
            p = p.buffer(0)
        return p if not p.is_empty else None
    except Exception:
        return None


def footprint_polygon(layers):
    """블록의 모든 레이어 폴리곤의 합집합(전체 평면 footprint)."""
    polys = [pp for pp in (_poly(l) for l in layers) if pp is not None]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else unary_union(polys)


# ---------------------------------------------------------------------------
# 장애물 / 배치 결과 자료구조
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    """이미 놓인 블록 하나 — footprint 와 bbox 를 캐시."""
    block: Block
    footprint: object
    bbox: tuple  # (min_x, min_y, max_x, max_y)
    area: float = 0.0


def make_obstacle(block: Block) -> Obstacle:
    fp = footprint_polygon(block.layers_at_pos())
    return Obstacle(block=block, footprint=fp,
                    bbox=block.bounding_rect(),
                    area=(fp.area if fp is not None else 0.0))


# 블록 footprint 면적 캐시: (id(instance), block_id) -> 면적 (orient 0 기준).
# 면적 사전검사(자유면적 부족 시 배치 시도 생략)에 사용.
_AREA_CACHE: dict = {}


def block_footprint_area(block_id: int, instance: dict) -> float:
    key = (id(instance), block_id)
    a = _AREA_CACHE.get(key)
    if a is None:
        og = _orient_geom(block_id, instance, 0)
        a = og.base_fp.area if og is not None else 0.0
        _AREA_CACHE[key] = a
    return a


@dataclass
class Placement:
    """배치 결과: 위치/방향 + 만들어진 Block + 패킹 품질 키."""
    block_id: int
    x: int
    y: int
    orient_idx: int
    block: Block
    top_y: float       # 배치 후 윗변 (낮을수록 좋음 = 바닥에 가까움)
    right_x: float     # 배치 후 우변 (낮을수록 좋음 = 왼쪽에 가까움)

    @property
    def key(self):
        # bottom-left 선호: 윗변이 낮고, 그다음 우변이 낮고, 그다음 좌하단
        return (self.top_y, self.right_x, self.x, self.y)


# ---------------------------------------------------------------------------
# 방향별 로컬 기하 캐시 (base footprint @ (0,0) + 로컬 bbox)
# ---------------------------------------------------------------------------
@dataclass
class _OrientGeom:
    base_fp: object                 # (0,0) 에 놓은 footprint
    lx0: float; ly0: float          # 로컬 min
    lx1: float; ly1: float          # 로컬 max


_GEOM_CACHE: dict = {}


def clear_cache() -> None:
    """인스턴스가 바뀔 때 호출(보통 불필요 — 인스턴스는 실행 중 불변)."""
    _GEOM_CACHE.clear()
    _AREA_CACHE.clear()


def _orient_geom(block_id: int, instance: dict, orient_idx: int) -> Optional[_OrientGeom]:
    # 인스턴스는 실행 중 불변이므로 (id, block, orient) 로 캐시
    key = (id(instance), block_id, orient_idx)
    cached = _GEOM_CACHE.get(key, False)
    if cached is not False:
        return cached
    g = _compute_orient_geom(block_id, instance, orient_idx)
    _GEOM_CACHE[key] = g
    return g


def _compute_orient_geom(block_id: int, instance: dict, orient_idx: int) -> Optional[_OrientGeom]:
    b0 = Block.from_instance(block_id, instance, x=0, y=0, orient_idx=orient_idx)
    fp = footprint_polygon(b0.layers_at_pos())
    if fp is None:
        return None
    bb = b0.bounding_rect()
    return _OrientGeom(base_fp=fp, lx0=bb[0], ly0=bb[1], lx1=bb[2], ly1=bb[3])


# ---------------------------------------------------------------------------
# 후보 위치 생성 (벽 + 장애물 bbox 모서리 + 폴리곤 꼭짓점)
# ---------------------------------------------------------------------------
def _candidate_edges(obstacles: list[Obstacle], bay: Bay,
                     vertex_level: bool) -> tuple[list[float], list[float]]:
    """새 블록의 좌변/하변이 닿을 수 있는 후보 x/y 모음."""
    xs = {0.0}
    ys = {0.0}
    for o in obstacles:
        x0, y0, x1, y1 = o.bbox
        xs.update((x0, x1)); ys.update((y0, y1))
        if vertex_level and o.footprint is not None:
            geoms = (o.footprint.geoms
                     if o.footprint.geom_type == "MultiPolygon"
                     else [o.footprint])
            for g in geoms:
                for vx, vy in g.exterior.coords:
                    xs.add(vx); ys.add(vy)
    xs = sorted(v for v in xs if -_TOL <= v <= bay.width + _TOL)
    ys = sorted(v for v in ys if -_TOL <= v <= bay.height + _TOL)
    return xs, ys


def _int_refs(edges: list[float], local_min: float,
              local_max: float, bound: float) -> list[int]:
    """좌(하)변 후보 -> 정수 기준점(reference) 좌표. 경계 안에 드는 것만."""
    refs = set()
    for e in edges:
        r = math.ceil(e - local_min - _TOL)       # 변을 e 에 맞추는 정수 ref
        if r + local_min >= -_TOL and r + local_max <= bound + _TOL:
            refs.add(int(r))
    return sorted(refs)


# ---------------------------------------------------------------------------
# 핵심: 배치 찾기
# ---------------------------------------------------------------------------
def find_placement_aabb(bay: Bay,
                        block_id: int,
                        instance: dict,
                        obstacles: list[Obstacle],
                        orient_indices: Optional[Iterable[int]] = None
                        ) -> Optional[Placement]:
    """
    고속 배치: 블록/장애물을 바운딩박스(AABB)로만 다뤄 순수 정수 구간 연산으로 배치.
    폴리곤 translate/교차가 전혀 없어 폴리곤 경로보다 10~100배 빠르다.

    AABB 비중첩은 footprint 비중첩보다 강한 조건이므로 항상 실현가능(안전영역).
    대신 오목한 틈을 못 써서 밀도는 낮다 -> 대규모/밀집 인스턴스에서
    '구성을 빨리 끝내 LNS 시간을 버는' 용도로 사용.
    """
    n_or = len(instance["blocks"][block_id]["shape"])
    if orient_indices is None:
        orient_indices = range(n_or)

    rects = [o.bbox for o in obstacles]
    # 후보 좌변 = 벽 + 장애물 우변; 후보 하변 = 바닥 + 장애물 윗변 (표준 bottom-left)
    xedges = {0.0}
    yedges = {0.0}
    for (x0, y0, x1, y1) in rects:
        xedges.add(x1)
        yedges.add(y1)

    best: Optional[Placement] = None
    for oi in orient_indices:
        og = _orient_geom(block_id, instance, oi)
        if og is None:
            continue
        if (og.lx1 - og.lx0) > bay.width + _TOL or (og.ly1 - og.ly0) > bay.height + _TOL:
            continue
        xs = _int_refs(sorted(xedges), og.lx0, og.lx1, bay.width)
        ys = _int_refs(sorted(yedges), og.ly0, og.ly1, bay.height)

        found = None
        for y in ys:
            cminy = y + og.ly0
            cmaxy = y + og.ly1
            for x in xs:
                cminx = x + og.lx0
                cmaxx = x + og.lx1
                ok = True
                for (bx0, by0, bx1, by1) in rects:
                    if (cmaxx <= bx0 + _TOL or bx1 <= cminx + _TOL
                            or cmaxy <= by0 + _TOL or by1 <= cminy + _TOL):
                        continue
                    ok = False
                    break
                if ok:
                    found = (x, y)
                    break
            if found:
                break

        if found is not None:
            x, y = found
            cand = Placement(
                block_id=block_id, x=x, y=y, orient_idx=oi,
                block=Block.from_instance(block_id, instance, x=x, y=y, orient_idx=oi),
                top_y=y + og.ly1, right_x=x + og.lx1,
            )
            if best is None or cand.key < best.key:
                best = cand
    return best


def find_placement(bay: Bay,
                   block_id: int,
                   instance: dict,
                   obstacles: list[Obstacle],
                   orient_indices: Optional[Iterable[int]] = None,
                   vertex_level: bool = True,
                   grid_step: int = 0) -> Optional[Placement]:
    """
    block_id 블록을 bay 에, obstacles 와 겹치지 않게 배치할 (x,y,방향)을 찾는다.

    bottom-left 우선: 각 방향마다 가장 낮고-왼쪽 자리를 찾고,
    방향들 사이에서 가장 빽빽한(윗변 낮은) 것을 고른다.

    Parameters
    ----------
    obstacles    : 이 블록과 시간이 겹치는, 이미 놓인 블록들의 Obstacle 목록.
    orient_indices : 시도할 방향(기본 전체).
    vertex_level : True 면 장애물 폴리곤 꼭짓점까지 후보로 -> 오목한 틈에 끼움.
    grid_step    : >0 이면 그 간격의 정수 격자도 후보에 추가(완전성 보강, 느려짐).

    Returns
    -------
    Placement | None
    """
    n_or = len(instance["blocks"][block_id]["shape"])
    if orient_indices is None:
        orient_indices = range(n_or)

    cand_xs, cand_ys = _candidate_edges(obstacles, bay, vertex_level)

    # 장애물 prepared 폴리곤 + bbox (빠른 교차 판정)
    prepared = [(o.bbox, _shp_prep(o.footprint))
                for o in obstacles if o.footprint is not None]

    best: Optional[Placement] = None

    for oi in orient_indices:
        og = _orient_geom(block_id, instance, oi)
        if og is None:
            continue
        bw = og.lx1 - og.lx0
        bh = og.ly1 - og.ly0
        if bw > bay.width + _TOL or bh > bay.height + _TOL:
            continue  # 이 방향으론 베이에 아예 안 들어감

        xs = _int_refs(cand_xs, og.lx0, og.lx1, bay.width)
        ys = _int_refs(cand_ys, og.ly0, og.ly1, bay.height)
        if grid_step > 0:
            xs = sorted(set(xs) | set(range(math.ceil(-og.lx0),
                                            int(bay.width - og.lx1) + 1, grid_step)))
            ys = sorted(set(ys) | set(range(math.ceil(-og.ly0),
                                            int(bay.height - og.ly1) + 1, grid_step)))

        found = _first_fit(og, xs, ys, prepared)
        if found is not None:
            x, y = found
            cand = Placement(
                block_id=block_id, x=x, y=y, orient_idx=oi,
                block=Block.from_instance(block_id, instance, x=x, y=y, orient_idx=oi),
                top_y=y + og.ly1, right_x=x + og.lx1,
            )
            if best is None or cand.key < best.key:
                best = cand

    return best


def _first_fit(og: _OrientGeom, xs: list[int], ys: list[int],
               prepared) -> Optional[tuple[int, int]]:
    """
    (y,x) 오름차순 bottom-left 로 첫 비중첩 위치를 찾아 즉시 반환.

    속도 최적화(품질은 동일):
      - 후보 AABB 가 어떤 장애물 AABB 와도 안 겹치면 -> 빈 공간, 폴리곤 연산 없이 즉시 채택.
      - 겹치는 장애물이 있을 때만 폴리곤 1회 translate + 그 몇 개와 교차검사.
      - 첫 적합에서 바로 반환(조기 종료) -> 빈 영역에선 거의 즉시 끝남.
    """
    for y in ys:
        cminy = y + og.ly0
        cmaxy = y + og.ly1
        for x in xs:
            cminx = x + og.lx0
            cmaxx = x + og.lx1
            # AABB 로 겹치는 장애물만 추림
            hits = None
            for (bx0, by0, bx1, by1), pgeom in prepared:
                if cmaxx <= bx0 + _TOL or bx1 <= cminx + _TOL:
                    continue
                if cmaxy <= by0 + _TOL or by1 <= cminy + _TOL:
                    continue
                if hits is None:
                    hits = []
                hits.append(pgeom)
            if hits is None:
                return (x, y)          # 빈 공간: 폴리곤 연산 없이 채택
            # 폴리곤 정밀 검사 (겹칠 가능성 있는 것만).
            # 충돌 = '내부-내부가 2D(면적)로 겹침'. 변끼리만 닿는 건 허용.
            # relate_pattern('2********') 은 기하 생성 없이 술어만 평가 -> 변접촉 빠르게 기각.
            cand_fp = _shp_translate(og.base_fp, xoff=x, yoff=y)
            ok = True
            for pgeom in hits:
                if pgeom.context.relate_pattern(cand_fp, "2********"):
                    ok = False
                    break
            if ok:
                return (x, y)
    return None


# ---------------------------------------------------------------------------
# 자가 테스트: 시간 무시하고 한 베이에 최대한 채워보며 엔진 검증
# ---------------------------------------------------------------------------
def _self_test(instance_path: str):
    import json
    inst = json.load(open(instance_path, encoding="utf-8"))
    bays = [Bay.from_dict(b, i) for i, b in enumerate(inst["bays"])]
    n_blocks = len(inst["blocks"])

    print(f"instance: {inst.get('name')}  | bays={len(bays)}  blocks={n_blocks}")
    for bay in bays:
        obstacles: list[Obstacle] = []
        placed = 0
        used_area = 0.0
        for bid in range(n_blocks):
            pl = find_placement(bay, bid, inst, obstacles)
            if pl is None:
                continue
            # 비중첩 재확인 (엔진 자체 검증)
            new_fp = footprint_polygon(pl.block.layers_at_pos())
            for o in obstacles:
                inter = new_fp.intersection(o.footprint)
                assert inter.is_empty or inter.area <= EPS_AREA, \
                    f"OVERLAP! block {bid} vs {o.block.block_id}"
            obstacles.append(make_obstacle(pl.block))
            placed += 1
            used_area += new_fp.area
        util = 100 * used_area / (bay.width * bay.height)
        print(f"  bay{bay.id} ({bay.width}x{bay.height}): "
              f"동시 배치 {placed}/{n_blocks}개, 면적 활용률 {util:.1f}%  (비중첩 검증 통과)")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "example_B2_b10.json"
    _self_test(path)
