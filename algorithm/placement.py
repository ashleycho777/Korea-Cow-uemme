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

import numpy as np

from utils import Block, Bay

from shapely.geometry import Polygon, box as _shp_box
from shapely.ops import unary_union
from shapely.affinity import translate as _aff_translate
import shapely as _shp

def _shp_translate(geom, xoff=0.0, yoff=0.0):
    """shapely.affinity.translate 대체 (동일 결과, 3~4배 빠름).

    affinity.translate 는 순수 파이썬 affine_transform 경로라 후보 위치마다
    numpy stack 을 새로 만든다. shapely.transform 은 C 레벨 좌표 변환이라 훨씬 싸다.
    폴리곤 배치 탐색의 최대 병목이 이 함수였다(프로파일 기준 전체의 59%).
    """
    return _shp.transform(geom, lambda c: c + (xoff, yoff))
from shapely.prepared import prep as _shp_prep
from shapely import relate_pattern as _relate, polygons
from shapely import area as _shp_area, intersection as _shp_inter

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
    coords: object = None           # 단순 Polygon 이면 (N,2) 좌표 배열 -> 벡터 경로 사용


_GEOM_CACHE: dict = {}


def clear_cache() -> None:
    """인스턴스가 바뀔 때 호출(보통 불필요 — 인스턴스는 실행 중 불변)."""
    _GEOM_CACHE.clear()
    _AREA_CACHE.clear()
    _MASK_CACHE.clear()


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
    # 구멍 없는 단순 Polygon 이면 좌표를 캐시해 벡터 경로를 쓴다.
    cds = None
    if fp.geom_type == "Polygon" and len(fp.interiors) == 0:
        cds = np.asarray(fp.exterior.coords, dtype=float)
    return _OrientGeom(base_fp=fp, lx0=bb[0], ly0=bb[1], lx1=bb[2], ly1=bb[3], coords=cds)


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
# ---------------------------------------------------------------------------
# 격자(래스터) 배치 엔진 — 보수적 래스터화 + numpy bottom-left
#   footprint 가 닿는 모든 1단위 셀을 마스크로 -> 격자 비중첩이면 footprint 비중첩 보장(feasible).
#   폴리곤을 매번 옮겨 교차하는 대신 numpy 불리언 연산 -> 훨씬 빠르고, 전수 BL 로 더 빽빽.
# ---------------------------------------------------------------------------
_MASK_CACHE: dict = {}


def block_mask(block_id: int, instance: dict, orient_idx: int):
    """(mask[h,w] bool, ox, oy) 반환. ref (0,0) 기준 셀 (i,j) 의 월드 좌하단 = (ox+i, oy+j)."""
    key = (id(instance), block_id, orient_idx)
    cached = _MASK_CACHE.get(key, 0)
    if cached != 0:
        return cached
    og = _orient_geom(block_id, instance, orient_idx)
    if og is None:
        _MASK_CACHE[key] = None
        return None
    ox = math.floor(og.lx0); oy = math.floor(og.ly0)
    w = int(math.ceil(og.lx1)) - ox
    h = int(math.ceil(og.ly1)) - oy
    if w <= 0 or h <= 0:
        _MASK_CACHE[key] = None
        return None
    cells = np.empty(h * w, dtype=object)
    k = 0
    for j in range(h):
        for i in range(w):
            cells[k] = _shp_box(ox + i, oy + j, ox + i + 1, oy + j + 1); k += 1
    mask = (_shp_area(_shp_inter(cells, og.base_fp)) > EPS_AREA).reshape(h, w)
    res = (mask, ox, oy)
    _MASK_CACHE[key] = res
    return res


def find_placement_grid(bay: Bay,
                        block_id: int,
                        instance: dict,
                        obstacles: list,
                        orient_indices: Optional[Iterable[int]] = None
                        ) -> Optional[Placement]:
    """격자 bottom-left 배치. 폴리곤 경로보다 빠르고, 전수 BL 로 더 빽빽. 항상 feasible."""
    W = int(round(bay.width)); H = int(round(bay.height))
    if W <= 0 or H <= 0:
        return None
    if orient_indices is None:
        orient_indices = range(len(instance["blocks"][block_id]["shape"]))

    occ = np.zeros((H, W), dtype=bool)
    for o in obstacles:
        b = o.block
        om = block_mask(b.block_id, instance, b.orient_idx)
        if om is None:
            continue
        m, ox, oy = om
        gx = int(round(b.x)) + ox; gy = int(round(b.y)) + oy
        mh, mw = m.shape
        x0 = max(0, gx); y0 = max(0, gy)
        x1 = min(W, gx + mw); y1 = min(H, gy + mh)
        if x1 <= x0 or y1 <= y0:
            continue
        occ[y0:y1, x0:x1] |= m[y0 - gy:y1 - gy, x0 - gx:x1 - gx]

    best = None
    for oi in orient_indices:
        bm = block_mask(block_id, instance, oi)
        if bm is None:
            continue
        m, mox, moy = bm
        mh, mw = m.shape
        if mw > W or mh > H:
            continue
        placed = None
        for gy0 in range(0, H - mh + 1):          # bottom-left: y 먼저
            band = occ[gy0:gy0 + mh, :]
            for gx0 in range(0, W - mw + 1):
                if not (band[:, gx0:gx0 + mw] & m).any():
                    placed = (gx0, gy0)
                    break
            if placed:
                break
        if placed:
            gx0, gy0 = placed
            og = _orient_geom(block_id, instance, oi)
            x = gx0 - mox; y = gy0 - moy
            key = (gy0 + mh, gx0)
            if best is None or key < best[0]:
                blk = Block.from_instance(block_id, instance, x=x, y=y, orient_idx=oi)
                best = (key, Placement(block_id=block_id, x=x, y=y, orient_idx=oi,
                                       block=blk, top_y=y + og.ly1, right_x=x + og.lx1))
    return best[1] if best else None


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
    # 낮은 방향부터 시도하면 좋은 top_y 를 일찍 잡아 가지치기가 강해진다.
    # (전 방향을 여전히 평가하므로 최종 선택 결과는 순서와 무관하게 동일하다.)
    _ogs = [(oi, _orient_geom(block_id, instance, oi)) for oi in orient_indices]
    _ogs = [t for t in _ogs if t[1] is not None]
    _ogs.sort(key=lambda t: t[1].ly1 - t[1].ly0)

    for oi, og in _ogs:
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
    # prep() 은 아래에서 .context(원본)로만 쓰여 준비 효과가 전혀 없었다 -> 제거.
    prepared = [(o.bbox, o.footprint)
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

        y_max = None if best is None else best.key[0] - og.ly1
        found = _first_fit(og, xs, ys, prepared, y_max)
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
               prepared, y_max: Optional[float] = None) -> Optional[tuple[int, int]]:
    """
    (y,x) 오름차순 bottom-left 로 첫 비중첩 위치를 찾아 즉시 반환.

    og.coords 가 있으면(단순 Polygon) 한 행(row)의 x 후보들을 청크 단위로 묶어
    shapely ufunc 으로 한 번에 판정한다. 판정 결과·반환 위치는 스칼라 경로와 동일하며
    파이썬 호출 오버헤드만 제거된다(폴리곤 판정 1건당 13.1us -> 1.7us).
    """
    if og.coords is None or not prepared:
        return _first_fit_scalar(og, xs, ys, prepared, y_max)

    obs_box = np.array([p[0] for p in prepared], dtype=float)          # (O,4)
    obs_geom = np.empty(len(prepared), dtype=object)
    for i, p in enumerate(prepared):
        obs_geom[i] = p[1]

    xa = np.asarray(xs, dtype=float)
    cminx_all = xa + og.lx0
    cmaxx_all = xa + og.lx1
    # x 축 AABB 겹침은 y 와 무관 -> 한 번만 계산
    ox0, oy0, ox1, oy1 = obs_box[:, 0], obs_box[:, 1], obs_box[:, 2], obs_box[:, 3]
    xov = (cmaxx_all[:, None] > ox0[None, :] + _TOL) & (ox1[None, :] > cminx_all[:, None] + _TOL)

    # 청크를 8부터 키운다: 앞쪽에서 바로 맞으면 낭비가 작고,
    # 계속 실패하는 빽빽한 행에서는 큰 청크로 벡터 이득을 최대한 받는다.
    CHUNKS = (8, 16, 32, 64, 128)
    for y in ys:
        if y_max is not None and y > y_max:
            break
        cminy = y + og.ly0
        cmaxy = y + og.ly1
        yov = (cmaxy > oy0 + _TOL) & (oy1 > cminy + _TOL)               # (O,)
        if not yov.any():
            return (xs[0], y)          # 이 행은 어떤 장애물과도 y 로 안 겹침
        ovl = xov & yov[None, :]                                        # (M,O)
        s = 0
        ci = 0
        M = len(xs)
        while s < M:
            e = min(s + CHUNKS[min(ci, len(CHUNKS) - 1)], M)
            ci += 1
            sub = ovl[s:e]
            free = ~sub.any(axis=1)
            if free.all():
                return (xs[s], y)
            hit_any = sub.any(axis=1)
            if not hit_any.any():
                return (xs[s], y)
            polys = polygons(og.coords[None, :, :] +
                             np.stack([xa[s:e], np.full(e - s, float(y))], axis=1)[:, None, :])
            # 장애물 단위로 돌면서, 이미 막힌 후보는 다시 검사하지 않는다.
            # (스칼라 경로의 '첫 충돌에서 break' 와 같은 절약을 벡터에서 재현)
            alive = np.ones(e - s, dtype=bool)
            cols = np.nonzero(sub.any(axis=0))[0]
            if len(cols) > 1:                       # 많이 막는 장애물부터
                cols = cols[np.argsort(-sub[:, cols].sum(axis=0))]
            for j in cols:
                sel = np.nonzero(alive & sub[:, j])[0]
                if len(sel) == 0:
                    continue
                bad = _relate(obs_geom[j], polys[sel], "2********")
                if bad.any():
                    alive[sel[bad]] = False
                    if not alive.any():
                        break
            ok = np.nonzero(alive)[0]
            if len(ok):
                return (xs[s + int(ok[0])], y)
            s = e
    return None


def _first_fit_scalar(og: _OrientGeom, xs: list[int], ys: list[int],
                      prepared, y_max: Optional[float] = None) -> Optional[tuple[int, int]]:
    """원래 경로(다중 폴리곤/구멍 있는 footprint 대비 폴백)."""
    for y in ys:
        if y_max is not None and y > y_max:
            break
        cminy = y + og.ly0
        cmaxy = y + og.ly1
        for x in xs:
            cminx = x + og.lx0
            cmaxx = x + og.lx1
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
                return (x, y)
            cand_fp = _shp_translate(og.base_fp, xoff=x, yoff=y)
            ok = True
            for pgeom in hits:
                if _relate(pgeom, cand_fp, "2********"):
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
