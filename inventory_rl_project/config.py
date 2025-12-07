# config.py: 환경, 비용 및 학습 하이퍼파라미터 설정
class Config:
    # 환경 파라미터
    MAX_CAPACITY = 200      # 최대 창고 용량
    INIT_INV = 50           # 초기 재고
    MAX_ORDER = 20          # 1회 최대 주문량 (정수)
    LEAD_TIME = 2           # 리드 타임 (기간 수)
    EPISODE_LENGTH = 100    # 에피소드 길이 (타임스텝)

    # 비용 파라미터
    HOLDING_COST = 1.0      # 단위당 재고 유지 비용
    STOCKOUT_COST = 10.0    # 단위당 재고 부족 비용
    ORDER_FIXED_COST = 5.0  # 주문 고정 비용
    ORDER_VAR_COST = 2.0    # 주문 단위당 변동 비용

    # 수요 분포 (포아송)
    DEMAND_MEAN = 5         # 일일 평균 수요

    # 학습 하이퍼파라미터 (PPO)
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    GAMMA = 0.99
    TOTAL_TIMESTEPS = 100_000
