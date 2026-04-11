"""
Fabio Valentini Pro Scalper Strategy
Converted from TradingView Pine Script v6

Order Flow Approximation for Scalping
Based on: Volume Absorption, Triple-A Setup, VWAP, Volume Profile, ORB
"""
import pandas as pd
import numpy as np
from typing import Any, Dict
from src.strategy import Strategy


class FabioValentiniProScalper(Strategy):
    """
    Fabio Valentini Pro Scalper Trading Strategy
    
    This strategy approximates order flow using volume analysis,
    absorption detection, triple-A setups (Absorption -> Accumulation -> Aggression),
    VWAP filtering, Opening Range Breakout, and Volume Profile levels.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        # Default parameters matching Pine Script inputs
        default_params = {
            # Session Settings
            'session_start': '0930',
            'session_end': '1600',
            'use_session_filter': True,
            
            # Volume Profile Settings
            'vp_length': 50,
            'vp_resolution': 24,
            
            # Absorption Detection
            'absorb_vol_mult': 2.0,
            'absorb_price_thresh': 0.3,
            
            # Delta Approximation
            'delta_lookback': 5,
            
            # VWAP Settings
            'use_vwap': True,
            'vwap_band': 0.5,
            
            # Risk Management
            'risk_percent': 1.0,
            'rr_ratio': 2.0,
            'max_daily_losses': 3,
            'trailing_stop': True,
            'trail_atr_mult': 1.5,
            
            # ORB Settings
            'orb_minutes': 30,
            'use_orb': True,
            
            # ATR
            'atr_period': 14
        }
        
        # Merge default parameters with user provided parameters
        if parameters:
            default_params.update(parameters)
            
        super().__init__(default_params)
        
        # State variables for persistence across bars
        self.state = {
            'daily_losses': 0,
            'last_session_date': None,
            'orb_high': np.nan,
            'orb_low': np.nan,
            'orb_defined': False,
            'absorption_count': 0
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV market data
        
        Args:
            data: DataFrame with columns: open, high, low, close, volume, datetime
            
        Returns:
            Series of trading signals (-1 = SHORT, 0 = HOLD, 1 = LONG)
        """
        df = data.copy()
        signals = pd.Series(0, index=df.index, name='signal')
        
        # === CALCULATIONS ===
        
        # ATR for dynamic levels
        df['atr'] = self._calculate_atr(df, self.parameters['atr_period'])
        
        # VWAP Calculation
        df['vwap'] = self._calculate_vwap(df)
        df['vwap_upper'] = df['vwap'] + df['atr'] * self.parameters['vwap_band']
        df['vwap_lower'] = df['vwap'] - df['atr'] * self.parameters['vwap_band']
        
        # Session Detection
        df['in_session'] = self._check_session(df)
        df['session_change'] = df['in_session'] & (~df['in_session'].shift(1).fillna(False))
        
        # Volume Profile (VAH, VAL, POC)
        df = self._calculate_volume_profile(df)
        
        # Delta Approximation (Buy vs Sell Pressure)
        df = self._calculate_delta(df)
        
        # Absorption Detection
        df = self._calculate_absorption(df)
        
        # Triple-A Setup Detection
        df = self._calculate_triple_a(df)
        
        # ORB (Opening Range Breakout)
        df = self._calculate_orb(df)
        
        # Value Area Bounce Setup
        df = self._calculate_value_area_bounces(df)
        
        # === COMBINED ENTRY SIGNALS ===
        
        # Long conditions
        df['long_signal'] = (
            (df['triple_a_long_setup'] | df['orb_breakout_long'] | df['val_bounce']) &
            (~self.parameters['use_vwap'] | (df['close'] > df['vwap_lower']))
        )
        
        # Short conditions
        df['short_signal'] = (
            (df['triple_a_short_setup'] | df['orb_breakout_short'] | df['vah_bounce']) &
            (~self.parameters['use_vwap'] | (df['close'] < df['vwap_upper']))
        )
        
        # Session and risk filter
        session_ok = ~self.parameters['use_session_filter'] | df['in_session']
        
        # Generate final signals
        for i in range(len(df)):
            if not self._can_trade(df.index[i]):
                signals.iloc[i] = 0
                continue
                
            if df['long_signal'].iloc[i] and session_ok.iloc[i]:
                signals.iloc[i] = 1
            elif df['short_signal'].iloc[i] and session_ok.iloc[i]:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0
        
        self.signals = signals
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (session based)"""
        # Group by date for daily VWAP
        df['date'] = df.index.date
        cum_vol = df.groupby('date')['volume'].cumsum()
        cum_pv = df.groupby('date').apply(
            lambda x: (x['close'] * x['volume']).cumsum()
        ).reset_index(level=0, drop=True)
        return cum_pv / cum_vol
    
    def _check_session(self, df: pd.DataFrame) -> pd.Series:
        """Check if bar is within trading session"""
        start_hour = int(self.parameters['session_start'][:2])
        start_min = int(self.parameters['session_start'][2:])
        end_hour = int(self.parameters['session_end'][:2])
        end_min = int(self.parameters['session_end'][2:])
        
        return df.index.to_series().apply(
            lambda dt: (dt.hour > start_hour or (dt.hour == start_hour and dt.minute >= start_min)) and
                       (dt.hour < end_hour or (dt.hour == end_hour and dt.minute <= end_min))
        )
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Profile: POC, VAH, VAL"""
        vp_length = self.parameters['vp_length']
        vp_resolution = self.parameters['vp_resolution']
        
        df['session_high'] = df['high'].rolling(window=vp_length).max()
        df['session_low'] = df['low'].rolling(window=vp_length).min()
        df['price_range'] = df['session_high'] - df['session_low']
        df['row_height'] = df['price_range'] / vp_resolution
        
        # Initialize VP columns
        df['poc'] = np.nan
        df['vah'] = np.nan
        df['val'] = np.nan
        
        # Calculate VP for each valid window
        for i in range(vp_length, len(df)):
            if np.isnan(df['price_range'].iloc[i]) or df['price_range'].iloc[i] <= 0:
                continue
                
            window = df.iloc[i-vp_length:i+1]
            session_low = df['session_low'].iloc[i]
            row_height = df['row_height'].iloc[i]
            
            # Volume at price array
            volume_at_price = np.zeros(vp_resolution)
            
            for _, bar in window.iterrows():
                close_val = float(bar['close'])
                session_low_val = float(session_low)
                row_height_val = float(row_height)
                price_level = int(np.floor((close_val - session_low_val) / row_height_val))
                if 0 <= price_level < vp_resolution:
                    volume_at_price[price_level] += float(bar['volume'])
            
            # Find POC (Point of Control)
            poc_index = np.argmax(volume_at_price)
            df.loc[df.index[i], 'poc'] = session_low + poc_index * row_height + (row_height / 2)
            
            # Calculate Value Area (70% volume)
            total_vol = volume_at_price.sum()
            target_vol = total_vol * 0.7
            
            accumulated_vol = volume_at_price[poc_index]
            vah_index = poc_index
            val_index = poc_index
            
            while accumulated_vol < target_vol and (vah_index < vp_resolution - 1 or val_index > 0):
                above_vol = volume_at_price[vah_index + 1] if vah_index < vp_resolution - 1 else 0
                below_vol = volume_at_price[val_index - 1] if val_index > 0 else 0
                
                if above_vol >= below_vol and vah_index < vp_resolution - 1:
                    vah_index += 1
                    accumulated_vol += above_vol
                elif val_index > 0:
                    val_index -= 1
                    accumulated_vol += below_vol
                else:
                    break
            
            df.loc[df.index[i], 'vah'] = session_low + vah_index * row_height + row_height
            df.loc[df.index[i], 'val'] = session_low + val_index * row_height
        
        return df
    
    def _calculate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Approximate Delta and buy/sell pressure"""
        lookback = self.parameters['delta_lookback']
        
        # Approximate buy/sell volume from candle body
        df['buy_volume'] = np.where(
            df['close'] > df['open'], df['volume'],
            np.where(df['close'] == df['open'], df['volume'] * 0.5,
                    df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low'] + 0.0001)))
        )
        df['sell_volume'] = df['volume'] - df['buy_volume']
        
        df['delta'] = df['buy_volume'] - df['sell_volume']
        df['smooth_delta'] = df['delta'].rolling(window=lookback).mean()
        df['cumulative_delta'] = df['delta'].cumsum()
        df['delta_momentum'] = df['cumulative_delta'].diff(lookback)
        
        # Delta control detection
        df['buyers_control'] = (df['smooth_delta'] > 0) & (df['smooth_delta'] > df['smooth_delta'].shift(1))
        df['sellers_control'] = (df['smooth_delta'] < 0) & (df['smooth_delta'] < df['smooth_delta'].shift(1))
        
        return df
    
    def _calculate_absorption(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume absorption patterns"""
        absorb_vol_mult = self.parameters['absorb_vol_mult']
        absorb_price_thresh = self.parameters['absorb_price_thresh']
        
        df['avg_volume'] = df['volume'].rolling(window=20).mean()
        df['price_move'] = abs(df['close'] - df['open']) / df['atr']
        df['high_volume'] = df['volume'] > df['avg_volume'] * absorb_vol_mult
        df['small_move'] = df['price_move'] < absorb_price_thresh
        
        df['absorption'] = df['high_volume'] & df['small_move']
        df['absorption_up'] = df['absorption'] & (df['close'] > df['open'])
        df['absorption_down'] = df['absorption'] & (df['close'] < df['open'])
        
        # Consecutive absorption count
        df['absorption_count'] = df['absorption'].astype(int).groupby(
            (~df['absorption']).cumsum()
        ).cumsum()
        df['strong_absorption'] = df['absorption_count'] >= 2
        
        return df
    
    def _calculate_triple_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Triple-A Setup: Absorption -> Accumulation -> Aggression"""
        # Accumulation detection (contraction)
        df['range_size'] = df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()
        df['avg_range'] = df['range_size'].rolling(window=50).mean()
        df['contraction'] = df['range_size'] < df['avg_range'] * 0.6
        
        # Aggression detection (expansion with volume)
        df['expansion'] = (df['range_size'] > df['avg_range'] * 1.2) & (df['volume'] > df['avg_volume'] * 1.5)
        df['aggressive_buy'] = df['expansion'] & (df['close'] > df['open']) & (df['smooth_delta'] > 0)
        df['aggressive_sell'] = df['expansion'] & (df['close'] < df['open']) & (df['smooth_delta'] < 0)
        
        # Triple-A setups (absorption within last 5 bars)
        df['triple_a_long'] = (
            df['absorption_up'].shift(3) |
            df['absorption_up'].shift(4) |
            df['absorption_up'].shift(5)
        )
        df['triple_a_long_setup'] = df['triple_a_long'] & df['contraction'].shift(1) & df['aggressive_buy']
        
        df['triple_a_short'] = (
            df['absorption_down'].shift(3) |
            df['absorption_down'].shift(4) |
            df['absorption_down'].shift(5)
        )
        df['triple_a_short_setup'] = df['triple_a_short'] & df['contraction'].shift(1) & df['aggressive_sell']
        
        return df
    
    def _calculate_orb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Opening Range Breakout signals"""
        orb_minutes = self.parameters['orb_minutes']
        
        df['session_bars'] = df.groupby(df.index.date).cumcount()
        
        # Calculate ORB high/low for each session
        def calc_orb(group):
            orb_window = group.head(int(orb_minutes / (df.index[1] - df.index[0]).seconds // 60))
            orb_high = orb_window['high'].max()
            orb_low = orb_window['low'].min()
            return pd.Series({'orb_high': orb_high, 'orb_low': orb_low})
        
        orb_values = df[df['in_session']].groupby(df.index.date).apply(calc_orb)
        df = df.merge(orb_values, left_on=df.index.date, right_index=True, how='left')
        
        df['orb_defined'] = df['session_bars'] >= (orb_minutes / (df.index[1] - df.index[0]).seconds // 60)
        
        # Breakout signals
        df['orb_breakout_long'] = self.parameters['use_orb'] & df['orb_defined'] & (df['close'] > df['orb_high']) & (df['close'].shift(1) <= df['orb_high'])
        df['orb_breakout_short'] = self.parameters['use_orb'] & df['orb_defined'] & (df['close'] < df['orb_low']) & (df['close'].shift(1) >= df['orb_low'])
        
        return df
    
    def _calculate_value_area_bounces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value area bounce signals"""
        # Price within 0.1% of VAL/VAH
        df['at_val'] = (~np.isnan(df['val'])) & (df['low'] <= df['val'] * 1.001) & (df['low'] >= df['val'] * 0.999)
        df['at_vah'] = (~np.isnan(df['vah'])) & (df['high'] >= df['vah'] * 0.999) & (df['high'] <= df['vah'] * 1.001)
        
        df['val_bounce'] = df['at_val'] & df['absorption_up'] & df['buyers_control']
        df['vah_bounce'] = df['at_vah'] & df['absorption_down'] & df['sellers_control']
        
        return df
    
    def _can_trade(self, current_datetime) -> bool:
        """Check if trading is allowed based on daily loss limit"""
        current_date = current_datetime.date()
        
        # Reset losses on new day
        if current_date != self.state['last_session_date']:
            self.state['daily_losses'] = 0
            self.state['last_session_date'] = current_date
        
        return self.state['daily_losses'] < self.parameters['max_daily_losses']
    
    def update_loss_counter(self, profit: float):
        """Update daily loss counter after a closed trade"""
        if profit < 0:
            self.state['daily_losses'] += 1
    
    def get_stop_loss_take_profit(self, side: int, price: float, atr: float) -> tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            side: 1 for LONG, -1 for SHORT
            price: entry price
            atr: current ATR value
            
        Returns:
            tuple (stop_loss, take_profit)
        """
        if side == 1:
            stop_loss = price - atr
            take_profit = price + (price - stop_loss) * self.parameters['rr_ratio']
        else:
            stop_loss = price + atr
            take_profit = price - (stop_loss - price) * self.parameters['rr_ratio']
            
        return stop_loss, take_profit
    
    def get_trailing_stop(self, atr: float) -> float:
        """Get trailing stop distance in price units"""
        if self.parameters['trailing_stop']:
            return atr * self.parameters['trail_atr_mult']
        return 0.0
