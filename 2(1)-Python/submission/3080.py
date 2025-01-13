from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        # 구현하세요!
        current_node = 0
        for element in seq:
            index = ord(element) - ord('A')  # 문자 -> 정수 인덱스로 변환

            # 이미 해당 index에 대한 공간이 없을 때만 추가
            if len(self[current_node].children) <= index or self[current_node].children[index] is None:
                # 필요한 만큼만 리스트를 확장
                while len(self[current_node].children) <= index:
                    self[current_node].children.append(None)
                # 새 노드 생성
                new_node_index = len(self)
                self.append(TrieNode())
                self[current_node].children[index] = new_node_index

            # 현재 노드를 갱신
            current_node = self[current_node].children[index]

        self[current_node].is_end = True  # 단어의 끝 표시
#####
    # 구현하세요!


import sys


"""
TODO:
- 일단 Trie부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    # 구현하세요!
    MOD = 1_000_000_007

    # 입력 처리
    input = sys.stdin.read
    data = input().splitlines()
    N = int(data[0])
    names = data[1:]

    # 이름을 사전순으로 정렬
    names.sort()

    # Trie 초기화
    trie = Trie()

    # 이름을 Trie에 삽입
    for name in names:
        trie.push(name)

    # 그룹별 조합 계산
    def count_combinations(node_index: int) -> int:
        """
        node_index: 현재 Trie 노드의 인덱스
        returns: 해당 노드에서 가능한 정렬 방법의 수
        """
        # 현재 노드에서 시작하는 이름의 수를 계산
        children = trie[node_index].children
        if not children:  # 자식 노드가 없으면 리프 노드
            return 1

        # 자식 노드들의 조합 계산
        combinations = 1
        for child_index in children:
            combinations *= count_combinations(child_index)
            combinations %= MOD

        # 현재 노드에서 가능한 모든 조합 반환
        return combinations * len(children) % MOD

    # 전체 가능한 조합 계산
    result = count_combinations(0)  # 루트 노드부터 시작

    # 결과 출력
    print(result)
    ####


if __name__ == "__main__":
    main()