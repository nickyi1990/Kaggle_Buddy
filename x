
strategy_config = {
    'name': 'DriftExposureLeverageStrategy_v2',
    'hold_period': '1H',

    'params': {
        # 初始策略权重分配
        # 注：每个子策略需为单方向的纯多、纯空策略
        # 按照strategy_pool中dict的顺序，依次对应权重和多空方向
        'cap_ratios': [0.5, 0.45, 0.05],
        'direction_list': ['long', 'short', 'short'],
        # 'pos_limit': {
        #     'monitor_strategy': '无脑空新币',
        #     'alt_ratios': [0.5, 0.5, 0, 0],
        # },
        # 初始化建仓时间（含该时刻）
        # 主要用于配置实盘初始资金分配时间，以便和回测保持一致
        # 要求：实盘初始时间 > 所有子策略回测的起始时间
        # 回测中保持start_date即可，实盘可指定具体时间，如：2026-03-01 12:20:00，并搭配对应的仓位比例cap_ratios
        # 初始时间和仓位比例从本地回测的data/子策略回测结果目录下的仓位比例-原始.csv中获取
        'init_time': '2025-01-01 00:00:00',

        # =========================
        # 敞口约束（常规）
        # =========================
        # 是否启用常规净敞口约束
        # 敞口 = (多头仓位 - 空头仓位) / 账户净值
        'exposure_limit_enabled': True,
        # 默认配置表示：最多允许偏空 10%（空头可比多头多 0.1）
        'exposure_limit_min': -0.1,
        # 默认配置表示：最多允许偏多 30%
        'exposure_limit_max': 0.3,

        # =========================
        # 二次敞口约束（低杠杆收紧）
        # =========================
        # 是否启用“低杠杆场景下的更严格敞口控制”
        'exposure_by_lev_enabled': True,
        # 触发阈值：当总杠杆 < 1.0 时触发
        # 触发后会额外应用更严格的敞口限制
        'exposure_by_lev_threshold': 1.0,
        # 默认配置表示：低杠杆下，敞口必须为0（即保持多空中性）
        'exposure_by_lev_min': -0.0,
        'exposure_by_lev_max': 0.0,

        # =========================
        # 总杠杆约束
        # =========================
        # 是否启用总杠杆上下限控制
        # 杠杆 = (多头仓位 + 空头仓位) / 账户净值
        'leverage_limit_enabled': True,
        # 默认配置表示：总杠杆不低于1x，不超过1.8x
        'leverage_limit_min': 1.0,
        'leverage_limit_max': 1.8,
    }
}

# 全部策略混合
strategy_pool = [
    # 0.5 = 都江堰0.15 + 四大龙狗0.15 + 纯合约双均线0.17 + 多1空1动量0.03
    dict(
        name='多头组合',
        strategy_list=[
            {
                "strategy": "Strategy_都江堰",
                "offset_list": [0],
                "hold_period": "1H",
                # "market": "mix_swap",
                "market": "swap_swap",
                'cap_weight': 0.15,
                'long_cap_weight': 1,
                'short_cap_weight': 0,
                'long_select_coin_num': 999,
                'short_select_coin_num': 0,
                "factor_list": [
                    ('QuoteVolumeMean', True, 24, 1),
                ],
                "filter_list": [
                    ('VolumeMaxBias_进出场阈值分离', (1203, 0.1, 0.7, start_date), 'val:==1'),
                ],
                "use_custom_func": False,
            },
            {
                "strategy": "Strategy_做多BTC",
                "offset_list": [0],
                "hold_period": "1H",
                "market": "swap_swap",
                'cap_weight': 0.15 / 4,
                'long_cap_weight': 1,
                'short_cap_weight': 0,
                'long_select_coin_num': 999,
                'short_select_coin_num': 0,
                "factor_list": [
                    ('SelectCoin', True, 'BTC-USDT', 1),
                ],
                "use_custom_func": False,
            },
            {
                "strategy": "Strategy_做多ETH",
                "offset_list": [0],
                "hold_period": "1H",
                "market": "swap_swap",
                'cap_weight': 0.15 / 4,
                'long_cap_weight': 1,
                'short_cap_weight': 0,
                'long_select_coin_num': 999,
                'short_select_coin_num': 0,
                "factor_list": [
                    ('SelectCoin', True, 'ETH-USDT', 1),
                ],
                "use_custom_func": False,
            },
            {
                "strategy": "Strategy_做多SOL",
                "offset_list": [0],
                "hold_period": "1H",
                "market": "swap_swap",
                'cap_weight': 0.15 / 4,
                'long_cap_weight': 1,
                'short_cap_weight': 0,
                'long_select_coin_num': 999,
                'short_select_coin_num': 0,
                "factor_list": [
                    ('SelectCoin', True, 'SOL-USDT', 1),
                ],
                "use_custom_func": False,
            },
            {
                "strategy": "Strategy_做多BNB",
                "offset_list": [0],
                "hold_period": "1H",
                "market": "swap_swap",
                'cap_weight': 0.15 / 4,
                'long_cap_weight': 1,
                'short_cap_weight': 0,
                'long_select_coin_num': 999,
                'short_select_coin_num': 0,
                "factor_list": [
                    ('SelectCoin', True, 'BNB-USDT', 1),
                ],
                "use_custom_func": False,
            },
        ]
    ),
    # 0.45 = 黄果树系列1 0.17 + 黄果树系列2 0.05 + 黄果树系列3 0.05 + 落单狗 0.1 + 马嵬坡 0.05 + 多1空1动量0.03
    dict(
        name='空头组合',
        strategy_list=[
            {
                "strategy": "Strategy_黄果树系列1",
                "offset_list": [16],
                "hold_period": "24H",
                "market": "swap_swap",
                'cap_weight': 0.17,
                'long_cap_weight': 0,
                'short_cap_weight': 1,
                'long_select_coin_num': 0,
                'short_select_coin_num': 0.5,
                "factor_list": [
                    ('Cci', False, 367, 1),
                ],
                "filter_list": [
                    ('QuoteVolumeMean', 367, 'pct:<0.2', False),
                    ('HoursSinceSpotAndSwap', 1, 'val:>0'),
                ],
                "use_custom_func": False,
            },
          ]
    ),
    # 落九天 0.05
    dict(
        name='落九天',
        strategy_list=[
            {
                "strategy": "Strategy_落九天",
                "offset_list": [0],
                "hold_period": "8H",
                "market": "swap_swap",
                'cap_weight': 0.5,
                'long_cap_weight': 0,
                'short_cap_weight': 1,
                'long_select_coin_num': 0,
                'short_select_coin_num': 999,
                "factor_list": [
                    ('ZfMeanQ', True, 60, 1),
                ],
                "filter_list": [
                    ('HoursSinceSpotAndSwap', 1, 'val:<585'),
                ],
                "filter_list_post": [
                    ('ZfMeanQ', 164, 'val:<0.5'),
                ],
                "use_custom_func": False,
            },
            {
                "strategy": "Strategy_落九天",
                "offset_list": [7],
                "hold_period": "8H",
                "market": "swap_swap",
                'cap_weight': 0.5,
                'long_cap_weight': 0,
                'short_cap_weight': 1,
                'long_select_coin_num': 0,
                'short_select_coin_num': 999,
                "factor_list": [
                    ('ZfMeanQ', True, 60, 1),
                ],
                "filter_list": [
                    ('HoursSinceSpotAndSwap', 1, 'val:<585'),
                ],
                "filter_list_post": [
                    ('ZfMeanQ', 397, 'val:<0.5'),
                ],
                "use_custom_func": False,
            }
        ],
    ),
]  # 策略池
