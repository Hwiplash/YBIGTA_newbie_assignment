from datasets import load_dataset

def load_corpus() -> list[str]:
    print("📢 데이터 로드 시작...")
    try:
        # 데이터셋 로드 시 로그 표시
        dataset = load_dataset(
            "google-research-datasets/poem_sentiment",
            split="train",
            cache_dir="C:/datasets_cache"
        )
        print("✅ 데이터셋 로드 완료")
        print(f"📢 데이터셋 컬럼: {dataset.column_names}")
        return [data["verse_text"] for data in dataset]
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return []
