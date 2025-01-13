from lib import Trie
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