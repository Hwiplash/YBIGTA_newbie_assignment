from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable, List

"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""

T = TypeVar("T")

@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: List[int] = field(default_factory=list)  # 수정
    is_end: bool = False

class Trie(Generic[T]):  # 리스트 상속 제거
    def __init__(self) -> None:
        self.nodes: List[TrieNode[T]] = [TrieNode[T](body=None)]  # 노드 리스트 사용

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 여러 번호 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        current: int = 0
        for element in seq:
            found = False
            for child_index in self.nodes[current].children:
                if self.nodes[child_index].body == element:
                    current = child_index
                    found = True
                    break

            if not found:
                new_node = TrieNode[T](body=element)  # **제네릭 타입 T 사용**
                self.nodes.append(new_node)
                self.nodes[current].children.append(len(self.nodes) - 1)
                current = len(self.nodes) - 1

        self.nodes[current].is_end = True

    def __getitem__(self, index: int) -> TrieNode[T]:
        """
        노드 접근 메서드
        """
        return self.nodes[index]
