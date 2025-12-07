# utils/visuals.py: 시뮬레이션 결과 시각화
import matplotlib.pyplot as plt

def plot_results(history):
    days = range(len(history['inv']))
    plt.figure(figsize=(12, 6))

    # 재고 변화 플롯
    plt.subplot(2, 1, 1)
    plt.plot(days, history['inv'], label='Inventory Level')
    plt.axhline(0, color='red', linestyle='--', label='Stockout Line')
    plt.title('Inventory Dynamics')
    plt.legend()

    # 주문량 및 수요 플롯
    plt.subplot(2, 1, 2)
    plt.bar(days, history['order'], label='Order Qty', alpha=0.6)
    plt.plot(days, history['demand'], 'r--', label='Demand')
    plt.title('Order vs Demand')
    plt.legend()

    plt.tight_layout()
    plt.show()
