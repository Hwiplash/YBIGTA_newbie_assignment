from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    # 세그먼트 트리 클래스: 구간합, 최솟값 등과 같은 범위 연산을 효율적으로 처리
    
    def __init__(self, n: int, default: T, merge: Callable[[T, T], T]):
        """
        초기화 함수
        :param n: 배열 크기
        :param default: 기본값 (초기 트리 값 및 범위 밖의 기본값)
        :param merge: 두 구간을 병합하는 함수
        """
        self.n = n  # 배열 크기
        self.default = default  # 기본값
        self.merge = merge  # 병합 함수
        self.tree: list[T] = [default for _ in range(4 * n)]  # 트리 배열 초기화

    def _build(self, arr: list[T], node: int, start: int, end: int):
        """
        세그먼트 트리 재귀적으로 구성
        :param arr: 초기 배열
        :param node: 현재 노드 인덱스
        :param start: 현재 구간의 시작 인덱스
        :param end: 현재 구간의 끝 인덱스
        """
        if start == end:  # 리프 노드인 경우
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self._build(arr, left_child, start, mid)
            self._build(arr, right_child, mid + 1, end)
            # 자식 노드 값을 병합하여 현재 노드 값 설정
            self.tree[node] = self.merge(self.tree[left_child], self.tree[right_child])

    def build(self, arr: list[T]):
        """
        주어진 배열로 세그먼트 트리를 구성
        :param arr: 초기 배열
        """
        self._build(arr, 0, 0, self.n - 1)

    def _update(self, node: int, start: int, end: int, idx: int, value: T):
        """
        세그먼트 트리 업데이트 (재귀 방식)
        :param node: 현재 노드 인덱스
        :param start: 현재 구간의 시작 인덱스
        :param end: 현재 구간의 끝 인덱스
        :param idx: 업데이트할 배열 인덱스
        :param value: 업데이트할 값
        """
        if start == end:  # 리프 노드인 경우
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            if idx <= mid:  # 업데이트할 인덱스가 왼쪽 자식 범위에 있는 경우
                self._update(left_child, start, mid, idx, value)
            else:  # 오른쪽 자식 범위에 있는 경우
                self._update(right_child, mid + 1, end, idx, value)
            # 자식 노드 값을 병합하여 현재 노드 값 갱신
            self.tree[node] = self.merge(self.tree[left_child], self.tree[right_child])

    def update(self, idx: int, value: T):
        """
        배열 특정 인덱스의 값을 업데이트
        :param idx: 업데이트할 배열 인덱스
        :param value: 업데이트할 값
        """
        self._update(0, 0, self.n - 1, idx, value)

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> T:
        """
        세그먼트 트리 구간 쿼리 (재귀 방식)
        :param node: 현재 노드 인덱스
        :param start: 현재 구간의 시작 인덱스
        :param end: 현재 구간의 끝 인덱스
        :param l: 쿼리 범위의 시작
        :param r: 쿼리 범위의 끝
        :return: 쿼리 결과
        """
        if r < start or l > end:  # 현재 구간이 쿼리 범위와 겹치지 않는 경우
            return self.default
        if l <= start and end <= r:  # 현재 구간이 쿼리 범위에 완전히 포함된 경우
            return self.tree[node]
        # 구간이 겹치는 경우, 왼쪽과 오른쪽 자식을 재귀적으로 탐색
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        left_result = self._query(left_child, start, mid, l, r)
        right_result = self._query(right_child, mid + 1, end, l, r)
        return self.merge(left_result, right_result)  # 결과 병합

    def query(self, l: int, r: int) -> T:
        """
        배열의 특정 구간 [l, r]에 대한 쿼리 수행
        :param l: 쿼리 범위의 시작
        :param r: 쿼리 범위의 끝
        :return: 쿼리 결과
        """
        return self._query(0, 0, self.n - 1, l, r)

####


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