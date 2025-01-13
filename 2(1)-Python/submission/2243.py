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

    n = int(data[0])  # 명령어 개수
    MAX_TASTE = 1_000_000  # 맛 점수의 최대값

    # 사탕 개수를 관리할 세그먼트 트리 초기화
    # 트리는 맛 점수별 사탕의 개수를 저장
    tree: SegmentTree[int, int] = SegmentTree(MAX_TASTE + 1, 0, lambda a, b: a + b)

    results = []  # 결과 저장

    for i in range(1, n + 1):
        query = list(map(int, data[i].split()))  # 명령어 읽기

        if query[0] == 1:  # 사탕 꺼내기 명령어
            rank = query[1]  # 꺼낼 사탕의 순위

            # 이분 탐색을 통해 꺼낼 사탕의 맛 점수 찾기
            low, high = 1, MAX_TASTE
            while low < high:
                mid = (low + high) // 2
                # [1, mid] 구간의 사탕 개수가 rank 이상이면 high를 줄임
                if tree.query(1, mid) >= rank:
                    high = mid
                else:  # 부족하면 low를 늘림
                    low = mid + 1

            results.append(low)  # 찾은 맛 점수를 결과에 추가
            # 해당 맛 점수의 사탕 개수를 1 감소
            tree.update(low, tree.query(low, low) - 1)

        elif query[0] == 2:  # 사탕 추가/제거 명령어
            taste = query[1]  # 맛 점수
            count = query[2]  # 추가(양수) 또는 제거(음수)할 사탕 개수
            current_count = tree.query(taste, taste)  # 현재 맛 점수의 사탕 개수 조회
            tree.update(taste, current_count + count)  # 사탕 개수 갱신

    # 모든 rank 쿼리 결과 출력
    sys.stdout.write("\n".join(map(str, results)) + "\n")

    ####


if __name__ == "__main__":
    main()