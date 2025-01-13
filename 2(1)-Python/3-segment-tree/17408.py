from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: 'Pair', b: 'Pair') -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([a[0], a[1], b[0], b[1]], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 입력 데이터 읽기
    import sys
    input = sys.stdin.read
    data = input().splitlines()

    # 배열 크기 N, 초기 배열 A, 쿼리 수 M
    N = int(data[0])
    A = list(map(int, data[1].split()))
    M = int(data[2])

    # 세그먼트 트리 초기화
    tree: SegmentTree[Pair, Pair] = SegmentTree(N, Pair.default(), Pair.f_merge)
    tree.build([Pair.f_conv(a) for a in A])

    results = []  # 결과 저장

    # 쿼리 처리
    for i in range(M):
        query = data[3 + i].split()

        if query[0] == "1":  # 값 갱신
            idx, value = int(query[1]), int(query[2])
            tree.update(idx - 1, Pair.f_conv(value))
        elif query[0] == "2":  # 구간합 계산
            l, r = int(query[1]), int(query[2])
            results.append(tree.query(l - 1, r - 1).sum())

    # 결과 출력
    sys.stdout.write("\n".join(map(str, results)) + "\n")


if __name__ == "__main__":
    main()