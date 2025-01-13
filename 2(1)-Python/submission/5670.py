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
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = None # 구현하세요!
        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                new_index = child_index
                break

        if new_index is None:
            break  # 정상적으로 trie에 등록된 단어이므로 여기서는 발생하지 않음

        pointer = new_index

    return cnt + 1  # 마지막 단어의 첫 글자 포함
####



def main() -> None:
    # 구현하세요!
    input = sys.stdin.read
    data = input().splitlines()

    while data:
        N = int(data[0])  # 단어의 개수
        words = data[1:1 + N]
        data = data[1 + N:]  # 다음 테스트 케이스로 이동

        # Trie 생성 및 단어 삽입
        trie = Trie()
        for word in words:
            trie.push(word)

        # 버튼 입력 횟수 계산
        total_presses = 0
        for word in words:
            total_presses += count(trie, word)

        # 평균 계산 및 출력
        average_presses = total_presses / N
        print(f"{average_presses:.2f}")


if __name__ == "__main__":
    main()