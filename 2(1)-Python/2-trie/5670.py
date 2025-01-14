from lib import Trie
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
