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



import sys
from typing import List, Optional

"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""

def count(trie: Trie[str], query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer: int = 0
    cnt: int = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index: Optional[int] = None
        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                new_index = child_index
                break

        if new_index is None:
            break  # 정상적으로 trie에 등록된 단어이므로 여기서는 발생하지 않음

        pointer = new_index

    return cnt + 1  # 마지막 단어의 첫 글자 포함

def main() -> None:
    input = sys.stdin.read
    data: List[str] = input().splitlines()

    while data:
        N: int = int(data[0])  # 단어의 개수
        words: List[str] = data[1:1 + N]
        data = data[1 + N:]  # 다음 테스트 케이스로 이동

        # Trie 생성 및 단어 삽입
        trie: Trie[str] = Trie()
        for word in words:
            trie.push(word)

        # 버튼 입력 횟수 계산
        total_presses: int = 0
        for word in words:
            total_presses += count(trie, word)

        # 평균 계산 및 출력
        average_presses: float = total_presses / N
        print(f"{average_presses:.2f}")

if __name__ == "__main__":
    main()
