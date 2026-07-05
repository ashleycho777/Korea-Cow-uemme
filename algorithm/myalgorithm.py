# myalgorithm.py — 대회 진입점 (이 파일명과 함수 시그니처는 고정)
# 실제 로직은 모듈로 분리:
#   placement.py   (배치 엔진)
#   constructor.py (그리드 구성 + 상태 API)
#   lns.py         (시간을 인자로 좋아지는 개선 루프)


def algorithm(prob_info, timelimit=60):
    """
    OGC 2026 알고리즘 진입점.
    prob_info : 문제 정보 dict, timelimit : 초 단위 시간 제한.
    반환 : {"operations": {...}} 형식의 해.
    """
    import lns
    return lns.solve(prob_info, timelimit)
