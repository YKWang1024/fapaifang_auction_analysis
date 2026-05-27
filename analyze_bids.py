import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def parse_price(price_str):
    return float(price_str.replace(',', ''))

def parse_time(time_str):
    time_str = time_str.strip()
    match = re.match(r'(\d{4})年(\d{2})月(\d{2})日\s+(\d{2}):(\d{2}):(\d{2})', time_str)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    return None

def analyze_bidding_data(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['状态', '竞买号', '价格', '时间'], encoding='gbk')
    df = df[df['竞买号'].notna() & (df['竞买号'] != '竞买号')]
    df['价格_num'] = df['价格'].apply(parse_price)
    df['时间_dt'] = df['时间'].apply(parse_time)
    df = df.sort_values('时间_dt', ascending=False).reset_index(drop=True)
    df['时间差_全局'] = df['时间_dt'].diff(-1).shift(-1)
    df['价格差_全局'] = df['价格_num'].diff(-1).shift(-1)
    df['价格差_全局_pct'] = (df['价格差_全局'] / df['价格_num'].shift(-1)) * 100
    bidder_stats = {}
    for bidder in df['竞买号'].unique():
        bidder_df = df[df['竞买号'] == bidder].copy().sort_values('时间_dt', ascending=False)
        bidder_df['时间差_自己'] = bidder_df['时间_dt'].diff(-1).shift(-1)
        bidder_df['价格差_自己'] = bidder_df['价格_num'].diff(-1).shift(-1)
        bidder_df['价格差_自己_pct'] = (bidder_df['价格差_自己'] / bidder_df['价格_num'].shift(-1)) * 100
        bidder_df['价格差_全局_pct'] = (bidder_df['价格差_全局'] / bidder_df['价格_num'].shift(-1)) * 100
        total_bids = len(bidder_df)
        initial_price = bidder_df.iloc[-1]['价格_num']
        max_price = bidder_df.iloc[0]['价格_num']
        avg_time_diff_global = bidder_df['时间差_全局'].dropna().mean()
        avg_time_diff_self = bidder_df['时间差_自己'].dropna().mean()
        avg_price_diff_global = bidder_df['价格差_全局'].dropna().mean()
        avg_price_diff_self = bidder_df['价格差_自己'].dropna().mean()
        avg_price_diff_global_pct = bidder_df['价格差_全局_pct'].dropna().mean()
        avg_price_diff_self_pct = bidder_df['价格差_自己_pct'].dropna().mean()
        shock_bids = (bidder_df['价格差_全局_pct'].dropna() > 3).sum()
        bidder_stats[bidder] = {
            '竞买号': bidder,
            '总竞价次数': total_bids,
            '初始竞价价格': initial_price,
            '最高竞价价格': max_price,
            '每次竞价相对最近一次出价的时间差的平均值(秒)': avg_time_diff_global.total_seconds() if pd.notna(avg_time_diff_global) else None,
            '每次竞价相对自己最近一次出价的时间差的平均值(秒)': avg_time_diff_self.total_seconds() if pd.notna(avg_time_diff_self) else None,
            '每次竞价相对最近一次出价的金额差的平均值': avg_price_diff_global,
            '每次竞价相对自己最近一次出价的金额差的平均值': avg_price_diff_self,
            '每次竞价相对最近一次出价的金额差与最近一次出价的百分比的平均值': avg_price_diff_global_pct,
            '每次竞价相对自己最近一次出价的金额差与总金额的百分比平均值': avg_price_diff_self_pct,
            '震慑性加价次数(>3%)': shock_bids
        }
    result_df = pd.DataFrame(bidder_stats.values())
    result_df = result_df.sort_values('最高竞价价格', ascending=False)
    return result_df

# 遍历data目录下的所有csv文件


filename_list = [
    f for f in os.listdir(r'd:\Working\法拍房竞价策略分析\data') if f.endswith('.csv')
]
for filename in filename_list:
    filename = filename.replace('.csv', '')
    csv_path = rf'd:\Working\法拍房竞价策略分析\data\{filename}.csv'
    try:
        result = analyze_bidding_data(csv_path)
        print("=" * 80)
        print(f"法拍房竞价策略分析结果 - {filename}")
        print("=" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(result.to_string(index=False))
        result.to_csv(rf'd:\Working\法拍房竞价策略分析\result\{filename}_竞价分析结果.csv', index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: d:\\Working\\法拍房竞价策略分析\\result\\{filename}_竞价分析结果.csv")
        print("\n")
    except Exception as e:
        print(f"处理文件 {filename}.csv 时出错: {e}")
        print("\n")