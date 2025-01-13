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


def main() -> None:
    # 입력 데이터 읽기
    input = sys.stdin.read
    data = input().splitlines()

    T = int(data[0])  # 테스트 케이스 수
    results = []  # 결과 저장

    for t in range(T):
        # 각 테스트 케이스에서 n: 영화 개수, m: 요청 횟수
        n, m = map(int, data[1 + t * 2].split())
        movies = list(map(int, data[2 + t * 2].split()))  # 요청된 영화 번호 리스트

        total_size = n + m  # 전체 트리 크기 계산 (영화 + 요청 구간)
        pos = [0] * (n + 1)  # 각 영화의 현재 위치 저장
        # 세그먼트 트리 초기화 (합 연산 트리)
        tree: SegmentTree[int, int] = SegmentTree(total_size + 1, 0, lambda a, b: a + b)

        # 초기 영화 위치 설정
        for i in range(1, n + 1):
            pos[i] = m + i  # 영화는 m 이후의 위치에 배치
            tree.update(pos[i], 1)  # 해당 위치에 영화 존재 표시

        current_top = m  # 요청된 영화들이 올라갈 위치의 시작점

        for movie in movies:
            # 현재 영화가 위치한 왼쪽 구간의 합(조회)
            results.append(tree.query(1, pos[movie] - 1))
            # 현재 영화 위치 제거
            tree.update(pos[movie], 0)
            # 영화의 새로운 위치 갱신
            pos[movie] = current_top
            # 새로운 위치에 영화 추가
            tree.update(current_top, 1)
            current_top -= 1  # 다음 요청된 영화가 올라갈 위치 이동

    # 결과 출력
    sys.stdout.write("\n".join(map(str, results)) + "\n")

    ####


if __name__ == "__main__":
    main()