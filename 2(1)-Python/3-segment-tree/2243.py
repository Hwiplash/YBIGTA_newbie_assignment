from lib import SegmentTree
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