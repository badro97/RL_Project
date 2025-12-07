# main.py: 환경 및 에이전트 생성과 학습/평가 루틴
from envs.inventory_env import InventoryEnv
from agents.ppo_agent import PPOInventoryAgent
from utils.visuals import plot_results
import numpy as np

def run_experiment(train=True, eval_episodes=1):
    # 환경과 에이전트 초기화
    env = InventoryEnv()
    agent = PPOInventoryAgent(env)

    # 학습 수행
    if train:
        agent.train()
        agent.save()

    # 평가 에피소드 수행
    all_histories = []
    for ep in range(eval_episodes):
        # Gymnasium의 reset() 반환값 처리 (obs, info)
        obs, info = env.reset()
        terminated = False
        truncated = False

        history = {'inv': [], 'demand': [], 'order': [], 'cost': []}

        # Gymnasium step 반환값(terminated, truncated) 처리
        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            history['inv'].append(info['inventory'])
            history['demand'].append(info['demand'])
            history['order'].append(info['order'])
            history['cost'].append(info['cost'])

        print(f"[Eval] Episode {ep+1} 총 비용: {sum(history['cost']):.2f}")
        all_histories.append(history)

    # 시각화 (첫 에피소드)
    plot_results(all_histories[0])

if __name__ == "__main__":
    run_experiment(train=True, eval_episodes=1)
