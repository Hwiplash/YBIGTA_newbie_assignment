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