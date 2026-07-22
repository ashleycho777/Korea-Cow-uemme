# myalgorithm.py
# 진입점: LNS 솔버(lns.py)를 호출한다.
# 실제 알고리즘 구현은 placement.py / constructor.py / lns.py 에 있음.


def algorithm(prob_info, timelimit=60):
    """
    대회 진입점. 시그니처를 변경하면 안 됨.
    prob_info : 문제 정보 dict
    timelimit : 초 단위 제한시간
    """
    import lns
    return lns.solve(prob_info, timelimit=timelimit)
