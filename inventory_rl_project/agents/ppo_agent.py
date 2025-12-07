# agents/ppo_agent.py: PPO 기반 재고 관리 에이전트
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import Config
import os

class PPOInventoryAgent:
    def __init__(self, env):
        # 단일 환경을 DummyVecEnv로 래핑 (벡터 환경 사용 권장)
        if not isinstance(env, DummyVecEnv):
            self.vec_env = DummyVecEnv([lambda: env])
        else:
            self.vec_env = env

        # PPO 모델 초기화
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            learning_rate=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
            batch_size=Config.BATCH_SIZE,
            n_steps=Config.N_STEPS,
        )

    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = Config.TOTAL_TIMESTEPS
        print("훈련 시작...")
        self.model.learn(total_timesteps=total_timesteps)
        print("훈련 완료.")

    def save(self, path="models/ppo_inventory_model"):
        # 모델 저장 (디렉토리 생성 포함)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        print(f"모델 저장 완료: {path}")

    def load(self, path="models/ppo_inventory_model"):
        self.model = PPO.load(path, env=self.vec_env)
        print(f"모델 로드 완료: {path}")

    def predict(self, state, deterministic=True):
        # 단일 상태에 대한 행동 예측 (PPO의 predict는 numpy array 또는 tensor 사용)
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)
