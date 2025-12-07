# envs/inventory_env.py: 재고 관리 환경 정의 (Gymnasium 호환)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config

class InventoryEnv(gym.Env):
    # 렌더링 모드 및 FPS 설정 (Gymnasium 렌더링 메타데이터)
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.capacity = Config.MAX_CAPACITY
        self.lead_time = Config.LEAD_TIME

        # 행동 공간: 0 ~ MAX_ORDER (정수)
        self.action_space = spaces.Discrete(Config.MAX_ORDER + 1)

        # 관측 공간: [현재 재고, 파이프라인 주문량...] (배열 길이 = 1+lead_time)
        low = np.array([-self.capacity] + [0] * self.lead_time, dtype=np.float32)
        high = np.array([self.capacity] + [Config.MAX_ORDER] * self.lead_time, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.day = 0
        self.seed_val = None

    def reset(self, seed=None, options=None):
        # 시드 설정 (Gymnasium 방식)
        if seed is not None:
            np.random.seed(seed)
            self.seed_val = seed
        self.day = 0
        current_inv = float(Config.INIT_INV)
        pipeline = [0 for _ in range(self.lead_time)]
        self.state = np.array([current_inv] + pipeline, dtype=np.float32)
        # Gymnasium reset: 상태와 info 반환
        return self.state, {}

    def step(self, action):
        current_inv = float(self.state[0])
        pipeline = list(self.state[1:])  # 파이프라인 주문량 리스트

        # 리드 타임 지난 주문 도착
        arriving_order = pipeline.pop(0) if self.lead_time > 0 else 0
        # 새로운 주문을 파이프라인 끝에 추가
        pipeline.append(int(action))
        # 입고 반영
        current_inv += float(arriving_order)

        # 확률적 수요 발생 (포아송 분포)
        demand = int(np.random.poisson(Config.DEMAND_MEAN))
        # 수요 충족 (백오더 허용)
        next_inv = current_inv - demand

        # 물리적 제약 (재고 상/하한)
        next_inv = min(next_inv, self.capacity)
        next_inv = max(next_inv, -self.capacity)

        # 비용 계산
        holding_cost = Config.HOLDING_COST * max(0.0, next_inv)
        stockout_cost = Config.STOCKOUT_COST * max(0.0, -next_inv)
        ordering_cost = 0.0
        if action > 0:
            ordering_cost = Config.ORDER_FIXED_COST + (Config.ORDER_VAR_COST * float(action))
        total_cost = holding_cost + stockout_cost + ordering_cost
        reward = -total_cost  # 보상은 비용의 음수

        # 상태 업데이트
        self.state = np.array([next_inv] + pipeline, dtype=np.float32)
        self.day += 1

        # 종료 여부: 기간 기반 종료 (terminated로 반환하고 truncated는 False)
        terminated = self.day >= Config.EPISODE_LENGTH
        truncated = False

        info = {
            "demand": demand,
            "order": int(action),
            "inventory": float(next_inv),
            "cost": float(total_cost)
        }

        # Gymnasium step: 상태, 보상, terminated, truncated, info 반환
        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        inv = self.state[0]
        pipeline = self.state[1:].tolist()
        print(f"Day {self.day} | Inv: {inv:.1f} | Pipeline: {pipeline}")

    def seed(self, seed=None):
        self.seed_val = seed
        np.random.seed(seed)
