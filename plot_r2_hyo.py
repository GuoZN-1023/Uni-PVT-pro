#!/usr/bin/env python3
import os

# ===================== 配置区域 =====================
DATA_DIR = "/home/thermo2025/Uni-PVT_results/results_hyo/分子编码组"
OUTPUT_FILE = os.path.join(DATA_DIR, "绘图数据.csv")

# ===================== 收集R2数据 =====================
def collect_r2_for_plotting():
    """收集所有R2数据，生成绘图用的CSV文件"""
    
    print("提取R2数据用于绘图...")
    print("="*60)
    
    # 存储结果
    results = []
    
    # 遍历所有实验
    for prop in ['Z', 'phi', 'H', 'S']:
        prop_dir = os.path.join(DATA_DIR, prop)
        
        if not os.path.exists(prop_dir):
            print(f"[{prop}] 目录不存在")
            continue
        
        for strat in ['001', '011', '111', '000']:
            exp_name = f"{prop}_PINN{strat}"
            exp_dir = os.path.join(prop_dir, exp_name)
            
            if not os.path.exists(exp_dir):
                continue
            
            # 找时间戳目录
            timestamp_dirs = []
            for item in os.listdir(exp_dir):
                item_path = os.path.join(exp_dir, item)
                if os.path.isdir(item_path) and item.startswith('202'):
                    timestamp_dirs.append(item_path)
            
            if not timestamp_dirs:
                continue
            
            # 用最新的目录
            latest_dir = max(timestamp_dirs, key=os.path.getmtime)
            metrics_file = os.path.join(latest_dir, 'exports', 'finetune_test_metrics.csv')
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        lines = f.readlines()
                    
                    # 第二行包含数据
                    if len(lines) >= 2:
                        data_line = lines[1].strip()  # 第二行
                        parts = data_line.split(',')
                        if len(parts) >= 3:  # stage, target, R2, ...
                            r2_value = float(parts[2])
                            results.append({
                                '物性': prop,
                                'PINN策略': strat,
                                'R2': r2_value,
                                '实验名称': exp_name
                            })
                            print(f"  {exp_name}: R² = {r2_value:.4f}")
                except Exception as e:
                    print(f"  {exp_name}: 读取失败 ({e})")
    
    return results

# ===================== 生成绘图数据文件 =====================
def create_plotting_data(results):
    """生成用于绘图的CSV文件"""
    
    if not results:
        print("❌ 没有找到任何R2数据")
        return None
    
    print(f"\n✅ 找到 {len(results)} 个实验结果")
    
    # 按物性分组的数据结构
    plot_data = {}
    for prop in ['Z', 'phi', 'H', 'S']:
        plot_data[prop] = {'001': None, '011': None, '111': None, '000': None}
    
    # 填充数据
    for item in results:
        prop = item['物性']
        strat = item['PINN策略']
        r2 = item['R2']
        plot_data[prop][strat] = r2
    
    # 生成CSV文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 标题行
        f.write("物性,PINN001,PINN011,PINN111,PINN000,最佳策略,最佳R2\n")
        
        for prop in ['Z', 'phi', 'H', 'S']:
            data = plot_data[prop]
            values = [data[s] for s in ['001', '011', '111', '000']]
            
            # 找最佳
            valid_values = [(s, v) for s, v in data.items() if v is not None]
            if valid_values:
                best_strat, best_r2 = max(valid_values, key=lambda x: x[1])
                # 写数据行
                f.write(f"{prop},{data['001'] or ''},{data['011'] or ''},{data['111'] or ''},{data['000'] or ''},{best_strat},{best_r2}\n")
            else:
                f.write(f"{prop},,,,,\n")
    
    print(f"\n💾 绘图数据已保存到: {OUTPUT_FILE}")
    
    # 显示数据预览
    print("\n📊 数据预览:")
    print("-" * 70)
    print(f"{'物性':<6} {'001':<8} {'011':<8} {'111':<8} {'000':<8} {'最佳':<6}")
    print("-" * 70)
    
    for prop in ['Z', 'phi', 'H', 'S']:
        data = plot_data[prop]
        valid_vals = [(s, v) for s, v in data.items() if v is not None]
        
        if valid_vals:
            best_strat, best_r2 = max(valid_vals, key=lambda x: x[1])
            print(f"{prop:<6} {data['001'] or 'N/A':<8.3f} {data['011'] or 'N/A':<8.3f} {data['111'] or 'N/A':<8.3f} {data['000'] or 'N/A':<8.3f} {best_strat:<6}")
        else:
            print(f"{prop:<6} N/A      N/A      N/A      N/A      N/A")
    
    print("-" * 70)
    
    return plot_data

# ===================== 生成Python绘图代码 =====================
def generate_plotting_code(plot_data, output_dir):
    """生成用于绘图的Python代码"""
    
    code_file = os.path.join(output_dir, "绘图代码.py")
    
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write('''#!/usr/bin/env python3
# 分子编码组R2对比图 - 绘图代码
import matplotlib.pyplot as plt
import numpy as np

# 数据（从 collect_r2_for_plotting.py 生成）
data = {
''')
        
        for prop in ['Z', 'phi', 'H', 'S']:
            prop_data = plot_data[prop]
            f.write(f"    '{prop}': {{\n")
            for strat in ['001', '011', '111', '000']:
                val = prop_data[strat]
                if val is not None:
                    f.write(f"        '{strat}': {val},\n")
                else:
                    f.write(f"        '{strat}': None,\n")
            f.write("    },\n")
        
        f.write('''}

# 设置
strategies = ['001', '011', '111', '000']
strategy_labels = ['001\\n(仅微调)', '011\\n(中后期)', '111\\n(全程)', '000\\n(无PINN)']
properties = ['Z', 'phi', 'H', 'S']
property_names = {
    'Z': '压缩因子 Z',
    'phi': '逸度系数 φ', 
    'H': '焓 H',
    'S': '熵 S'
}

# 颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 创建子图
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for idx, (prop, ax) in enumerate(zip(properties, axes)):
    prop_data = data[prop]
    
    # 检查是否有数据
    has_data = any(v is not None for v in prop_data.values())
    
    if not has_data:
        ax.text(0.5, 0.5, f'无{prop}数据', ha='center', va='center')
        ax.set_title(property_names.get(prop, prop))
        continue
    
    # 准备数据
    x_pos = np.arange(4)
    heights = [prop_data[s] if prop_data[s] is not None else 0 for s in strategies]
    
    # 绘制柱状图
    bars = ax.bar(x_pos, heights, color=colors, edgecolor='black', linewidth=1)
    
    # 标注数值
    for bar, height in zip(bars, heights):
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(property_names.get(prop, prop), fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 高亮最佳
    best_idx = np.argmax(heights)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

# 设置y轴标签
axes[0].set_ylabel('R²值', fontsize=11)

# 总标题
plt.suptitle('分子编码模型：PINN策略性能对比', fontsize=14, fontweight='bold', y=1.05)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('分子编码_R2对比图.png', dpi=300, bbox_inches='tight')
plt.show()

print("图片已保存为: 分子编码_R2对比图.png")
''')
    
    print(f"📝 绘图代码已生成: {code_file}")
    print("\n运行方法:")
    print(f"  python {code_file}")

# ===================== 主程序 =====================
def main():
    print("分子编码组R2数据提取")
    print("="*60)
    
    # 1. 收集数据
    results = collect_r2_for_plotting()
    
    # 2. 生成绘图数据文件
    plot_data = create_plotting_data(results)
    
    if plot_data is None:
        return
    
    # 3. 生成绘图代码
    generate_plotting_code(plot_data, DATA_DIR)
    
    # 4. 完成
    print("\n" + "="*60)
    print("✅ 完成！")
    print(f"1. 数据文件: {OUTPUT_FILE}")
    print("2. 绘图代码: 绘图代码.py")
    print("\n📊 下一步:")
    print("   在有matplotlib的环境中运行 '绘图代码.py' 生成图片")
    print("="*60)

# ===================== 运行 =====================
if __name__ == "__main__":
    main()