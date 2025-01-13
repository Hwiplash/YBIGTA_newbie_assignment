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