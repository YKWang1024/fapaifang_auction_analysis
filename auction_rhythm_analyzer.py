import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def parse_price(price_str):
    return float(str(price_str).replace(',', ''))

def parse_time(time_str):
    time_str = str(time_str).strip()
    match = re.match(r'(\d{4})年(\d{2})月(\d{2})日\s+(\d{2}):(\d{2}):(\d{2})', time_str)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return None

def analyze_auction_rhythm(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['状态', '竞买号', '价格', '时间'], encoding='gbk', on_bad_lines='skip')
    df = df[df['竞买号'].notna() & (df['竞买号'] != '竞买号')]
    df = df[df['竞买号'].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0)]
    df['价格_num'] = df['价格'].apply(parse_price)
    df['时间_dt'] = df['时间'].apply(parse_time)
    df = df.dropna(subset=['时间_dt', '价格_num'])
    df = df.sort_values('时间_dt', ascending=True).reset_index(drop=True)

    df['序号'] = range(1, len(df) + 1)
    df['时间差_秒'] = df['时间_dt'].diff().dt.total_seconds()
    df['价格差'] = df['价格_num'].diff()
    df['价格差_pct'] = (df['价格差'] / df['价格_num'].shift(1)) * 100
    df['累计加价'] = df['价格_num'] - df['价格_num'].iloc[0]
    df['累计加价_pct'] = (df['累计加价'] / df['价格_num'].iloc[0]) * 100

    auction_duration = (df['时间_dt'].iloc[-1] - df['时间_dt'].iloc[0]).total_seconds()

    n = len(df)
    early_n = max(3, n // 4)
    mid_start = n // 4
    mid_end = 3 * n // 4
    late_n = max(3, n // 4)

    early_df = df.head(early_n)
    mid_df = df.iloc[mid_start:mid_end]
    late_df = df.tail(late_n)

    stats = {
        '基本信息': {
            '总竞价次数': n,
            '竞拍总时长(分钟)': round(auction_duration / 60, 2),
            '起始价': df['价格_num'].iloc[0],
            '最终价': df['价格_num'].iloc[-1],
            '总加价额': df['价格_num'].iloc[-1] - df['价格_num'].iloc[0],
            '总加价百分比': round((df['价格_num'].iloc[-1] - df['价格_num'].iloc[0]) / df['价格_num'].iloc[0] * 100, 2),
            '参与竞买人数': df['竞买号'].nunique(),
            '平均每次竞价时间间隔(秒)': round(df['时间差_秒'].mean(), 2) if n > 1 else 0,
            '平均每次加价金额': round(df['价格差'].mean(), 2) if n > 1 else 0,
        },
        '节奏分析': {
            '前期平均时间间隔(秒)': round(early_df['时间差_秒'].mean(), 2) if len(early_df) > 1 else 0,
            '中期平均时间间隔(秒)': round(mid_df['时间差_秒'].mean(), 2) if len(mid_df) > 1 else 0,
            '后期平均时间间隔(秒)': round(late_df['时间差_秒'].mean(), 2) if len(late_df) > 1 else 0,
            '前期平均加价金额': round(early_df['价格差'].mean(), 2) if len(early_df) > 1 else 0,
            '中期平均加价金额': round(mid_df['价格差'].mean(), 2) if len(mid_df) > 1 else 0,
            '后期平均加价金额': round(late_df['价格差'].mean(), 2) if len(late_df) > 1 else 0,
            '节奏变化_时间': '变快' if late_df['时间差_秒'].mean() < early_df['时间差_秒'].mean() else '变慢',
            '节奏变化_幅度': '变小' if late_df['价格差'].mean() < early_df['价格差'].mean() else '变大',
        },
        '极值分析': {
            '最长等待时间(秒)': round(df['时间差_秒'].max(), 2) if n > 1 else 0,
            '最短等待时间(秒)': round(df['时间差_秒'].min(), 2) if n > 1 else 0,
            '最大单次加价': round(df['价格差'].max(), 2) if n > 1 else 0,
            '最小单次加价': round(df['价格差'].min(), 2) if n > 1 else 0,
            '最大加价百分比': round(df['价格差_pct'].max(), 4) if n > 1 else 0,
            '最后5次平均时间间隔(秒)': round(df.tail(5)['时间差_秒'].mean(), 2) if n >= 5 else 0,
            '最后5次平均加价金额': round(df.tail(5)['价格差'].mean(), 2) if n >= 5 else 0,
        },
        '最后赢家': df.iloc[-1]['竞买号'] if n > 0 else 'N/A'
    }

    bidder_summary = df.groupby('竞买号').agg({
        '价格_num': ['first', 'last', 'count'],
        '时间_dt': ['first', 'last']
    }).reset_index()
    bidder_summary.columns = ['竞买号', '初始出价', '最终出价', '出价次数', '首次出价时间', '末次出价时间']
    bidder_summary['加价总额'] = bidder_summary['最终出价'] - bidder_summary['初始出价']
    bidder_summary['参与时长(秒)'] = (bidder_summary['末次出价时间'] - bidder_summary['首次出价时间']).dt.total_seconds()
    bidder_summary = bidder_summary.sort_values('最终出价', ascending=False)

    return df, stats, bidder_summary

def create_rhythm_charts(df, stats, bidder_summary, output_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'竞拍节奏分析 - {Path(output_path).stem}', fontsize=16, fontweight='bold')

    bidder_order = bidder_summary['竞买号'].tolist()
    n_bidders = len(bidder_order)
    if n_bidders <= 10:
        cmap = plt.cm.get_cmap('tab10')
    elif n_bidders <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('hsv', n_bidders)
    colors_map = {bidder: cmap(i / max(n_bidders - 1, 1))
                  for i, bidder in enumerate(bidder_order)}

    ax1 = axes[0, 0]
    for bidder in bidder_order:
        bidder_df = df[df['竞买号'] == bidder]
        ax1.scatter(bidder_df['序号'], bidder_df['价格_num'] / 10000,
                    c=[colors_map[bidder]], label=bidder, s=50, zorder=5)
    ax1.plot(df['序号'], df['价格_num'] / 10000, 'gray', linewidth=1, alpha=0.5, linestyle='--', zorder=3)
    ax1.set_xlabel('竞价序号')
    ax1.set_ylabel('价格(万元)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)

    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(df['序号'], df['时间差_秒'].fillna(0) / 60, alpha=0.3, color='orange', label='时间间隔')
    ax1_twin.set_ylabel('时间间隔(分钟)', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.set_title('价格与时间间隔变化趋势 (按竞买人着色)')

    ax2 = axes[0, 1]
    bars = ax2.barh(bidder_summary['竞买号'], bidder_summary['出价次数'],
                    color=[colors_map[b] for b in bidder_summary['竞买号']])
    for i, (bar, final_price) in enumerate(zip(bars, bidder_summary['最终出价'])):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{final_price/10000:.1f}万', va='center', fontsize=9)
    ax2.set_xlabel('出价次数')
    ax2.set_title('各竞买人出价次数与最终出价')

    ax3 = axes[1, 0]
    rhythm_colors = ['green' if x > 0 else 'red' for x in df['价格差_pct'].fillna(0)]
    ax3.bar(df['序号'][1:], df['价格差_pct'][1:], color=rhythm_colors[1:], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=3, color='red', linestyle='--', linewidth=1.5, label='震慑线(3%)')
    ax3.set_xlabel('竞价序号')
    ax3.set_ylabel('加价百分比(%)')
    ax3.set_title('每次加价幅度百分比 (震慑线3%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    window = min(5, max(3, len(df) // 5))
    df['移动平均_时间'] = df['时间差_秒'].rolling(window=window, min_periods=1).mean() / 60
    df['移动平均_价格'] = df['价格差'].rolling(window=window, min_periods=1).mean()

    ax4.plot(df['序号'], df['移动平均_时间'], 'orange', linewidth=2, label=f'{window}次移动平均时间')
    ax4.set_xlabel('竞价序号')
    ax4.set_ylabel('平均时间间隔(分钟)', color='orange')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax4.grid(True, alpha=0.3)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(df['序号'], df['移动平均_价格'] / 10000, 'purple', linewidth=2, label=f'{window}次移动平均金额')
    ax4_twin.set_ylabel('平均加价金额(万元)', color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    ax4.set_title(f'节奏趋势 ({window}次移动平均)')

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax5 = axes[2, 0]
    phases = ['前期', '中期', '后期']
    time_means = [
        stats['节奏分析']['前期平均时间间隔(秒)'] / 60,
        stats['节奏分析']['中期平均时间间隔(秒)'] / 60,
        stats['节奏分析']['后期平均时间间隔(秒)'] / 60
    ]
    price_means = [
        stats['节奏分析']['前期平均加价金额'] / 10000,
        stats['节奏分析']['中期平均加价金额'] / 10000,
        stats['节奏分析']['后期平均加价金额'] / 10000
    ]
    x = np.arange(len(phases))
    width = 0.35
    bars1 = ax5.bar(x - width/2, time_means, width, label='平均时间间隔(分钟)', color='steelblue')
    ax5.set_ylabel('时间间隔(分钟)', color='steelblue')
    ax5.tick_params(axis='y', labelcolor='steelblue')
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, price_means, width, label='平均加价金额(万元)', color='coral')
    ax5_twin.set_ylabel('加价金额(万元)', color='coral')
    ax5_twin.tick_params(axis='y', labelcolor='coral')
    ax5.set_xticks(x)
    ax5.set_xticklabels(phases)
    ax5.set_title('竞拍阶段节奏对比')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')

    ax6 = axes[2, 1]
    ax6.axis('off')
    info_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      竞拍节奏统计摘要                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  总竞价次数: {stats['基本信息']['总竞价次数']:<8}    竞拍总时长: {stats['基本信息']['竞拍总时长(分钟)']:.1f}分钟          ║
    ║  起始价: {stats['基本信息']['起始价']/10000:.1f}万    最终价: {stats['基本信息']['最终价']/10000:.1f}万              ║
    ║  总加价额: {stats['基本信息']['总加价额']/10000:.1f}万    加价比例: {stats['基本信息']['总加价百分比']:.1f}%           ║
    ║  参与人数: {stats['基本信息']['参与竞买人数']:<4}    平均出价: {stats['基本信息']['平均每次加价金额']/10000:.2f}万/次    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                        节奏特征判断                            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  节奏变化(时间): {stats['节奏分析']['节奏变化_时间']:<6}    节奏变化(幅度): {stats['节奏分析']['节奏变化_幅度']:<6}      ║
    ║  前期节奏: {stats['节奏分析']['前期平均时间间隔(秒)']:.1f}秒/{stats['节奏分析']['前期平均加价金额']/10000:.2f}万                                    ║
    ║  后期节奏: {stats['节奏分析']['后期平均时间间隔(秒)']:.1f}秒/{stats['节奏分析']['后期平均加价金额']/10000:.2f}万                                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  最长等待: {stats['极值分析']['最长等待时间(秒)']:.0f}秒    最短等待: {stats['极值分析']['最短等待时间(秒)']:.1f}秒                        ║
    ║  最大单次加价: {stats['极值分析']['最大单次加价']/10000:.2f}万    震慑性加价: {'有' if stats['极值分析']['最大加价百分比'] > 3 else '无'}        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  最终赢家: {stats['最后赢家']:<10}                                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    ax6.text(0.5, 0.5, info_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图表已保存: {output_path}")

def process_all_csv_files():
    data_dir = Path(r'd:\Working\法拍房竞价策略分析\data')
    result_dir = Path(r'd:\Working\法拍房竞价策略分析\result')

    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print("未找到CSV文件")
        return

    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"处理文件: {csv_file.name}")
        print('='*60)

        try:
            df, stats, bidder_summary = analyze_auction_rhythm(str(csv_file))

            print("\n【基本信息】")
            for key, value in stats['基本信息'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")

            print("\n【节奏分析】")
            for key, value in stats['节奏分析'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")

            print("\n【极值分析】")
            for key, value in stats['极值分析'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value}")

            print(f"\n>> 最终赢家: {stats['最后赢家']}")

            chart_name = csv_file.stem + '_节奏分析图.png'
            chart_path = result_dir / chart_name
            create_rhythm_charts(df, stats, bidder_summary, str(chart_path))

        except Exception as e:
            print(f"处理文件出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    process_all_csv_files()