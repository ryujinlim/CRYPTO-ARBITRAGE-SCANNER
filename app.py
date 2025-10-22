
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Crypto Arbitrage Scanner",
    layout="wide",
)

st.title("Crypto Arbitrage Scanner")
st.caption(
    "Track real-time and historical price differences for major crypto assets across"
    " multiple exchanges."
)

SUPPORTED_EXCHANGES: Dict[str, str] = {
    "kucoin": "KuCoin", 
    "kraken": "Kraken",
    "coinbase": "Coinbase Pro",
    "bybit": "Bybit",
}

SUPPORTED_ASSETS: Dict[str, str] = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "DOGE": "DOGE/USDT",
}

TIMEFRAME_OPTIONS: Dict[str, str] = {
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d",
}

DEFAULT_ASSETS: Tuple[str, ...] = ("BTC", "ETH")

# Default transaction costs (fallback if real-time fetching fails)
DEFAULT_TRANSACTION_COSTS = {
    "kucoin": 0.1,  # 0.1% maker/taker
    "kraken": 0.16,  # 0.16% maker/taker
    "coinbase": 0.5,  # 0.5% maker/taker
    "bybit": 0.1,  # 0.1% maker/taker
}

# Minimum profit threshold (in USD)
MIN_PROFIT_THRESHOLD = 10.0


@st.cache_resource(show_spinner=False)
def get_exchange_client(exchange_id: str) -> ccxt.Exchange:
    """Instantiate and cache an exchange client."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    
    try:
        exchange.load_markets()
    except Exception as e:
        # Handle geographic restrictions and other exchange availability issues
        error_msg = str(e)
        if "restricted location" in error_msg.lower() or "service unavailable" in error_msg.lower():
            raise ccxt.ExchangeNotAvailable(f"{SUPPORTED_EXCHANGES[exchange_id]} is not available in your region. Please try other exchanges.")
        else:
            raise e
    
    return exchange


@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_exchange_fees(exchange_id: str, symbol: str) -> Dict[str, float]:
    """Fetch real-time trading fees from exchange."""
    try:
        exchange = get_exchange_client(exchange_id)
        
        # Try to fetch trading fees for the specific symbol
        if hasattr(exchange, 'fetch_trading_fee'):
            try:
                fee_info = exchange.fetch_trading_fee(symbol)
                if fee_info and 'taker' in fee_info:
                    return {
                        'maker': fee_info.get('maker', 0.001),
                        'taker': fee_info.get('taker', 0.001)
                    }
            except:
                pass
        
        # Try to fetch all trading fees
        if hasattr(exchange, 'fetch_trading_fees'):
            try:
                all_fees = exchange.fetch_trading_fees()
                if all_fees and symbol in all_fees:
                    fee_info = all_fees[symbol]
                    return {
                        'maker': fee_info.get('maker', 0.001),
                        'taker': fee_info.get('taker', 0.001)
                    }
            except:
                pass
                
    except Exception:
        pass
    
    # Fallback to default fees
    default_rate = DEFAULT_TRANSACTION_COSTS.get(exchange_id, 0.1) / 100
    return {
        'maker': default_rate,
        'taker': default_rate
    }


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_price(exchange_id: str, symbol: str) -> Optional[float]:
    """Fetch the latest trade price for a symbol on an exchange."""
    try:
        exchange = get_exchange_client(exchange_id)
        if symbol not in exchange.symbols:
            return None

        ticker = exchange.fetch_ticker(symbol)
        price_candidates = [
            ticker.get("last"),
            ticker.get("close"),
            ticker.get("ask"),
            ticker.get("bid"),
        ]
        price = next((value for value in price_candidates if value and not math.isnan(value)), None)
        return price
    except ccxt.ExchangeNotAvailable:
        # Exchange is not available in this region, return None
        return None
    except Exception:
        # Other errors, return None
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_historical_prices(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since: int,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a trading pair."""
    try:
        exchange = get_exchange_client(exchange_id)
        if symbol not in exchange.symbols:
            return pd.DataFrame()

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    except ccxt.ExchangeNotAvailable:
        # Exchange is not available in this region, return empty DataFrame
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive logging for Streamlit UI
        st.warning(f"{SUPPORTED_EXCHANGES[exchange_id]}: {exc}")
        return pd.DataFrame()

    if not ohlcv:
        return pd.DataFrame()

    frame = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    frame["exchange"] = SUPPORTED_EXCHANGES[exchange_id]
    return frame


def calculate_arbitrage_profit(
    buy_price: float, 
    sell_price: float, 
    buy_exchange: str, 
    sell_exchange: str, 
    symbol: str,
    amount_usd: float = 1000.0
) -> Dict[str, float]:
    """Calculate arbitrage profit including real-time transaction costs."""
    if not buy_price or not sell_price:
        return {"net_profit": 0, "gross_profit": 0, "buy_fee": 0, "sell_fee": 0, "roi": 0, "buy_fee_rate": 0, "sell_fee_rate": 0}
    
    # Get real-time fees from exchanges
    buy_fees = get_exchange_fees(buy_exchange, symbol)
    sell_fees = get_exchange_fees(sell_exchange, symbol)
    
    # Use taker fees (more realistic for arbitrage)
    buy_fee_rate = buy_fees.get('taker', 0.001)
    sell_fee_rate = sell_fees.get('taker', 0.001)
    
    # Calculate gross profit
    gross_profit = (sell_price - buy_price) * (amount_usd / buy_price)
    
    # Calculate fees
    buy_fee = amount_usd * buy_fee_rate
    sell_fee = (amount_usd + gross_profit) * sell_fee_rate
    
    # Calculate net profit
    net_profit = gross_profit - buy_fee - sell_fee
    
    # Calculate ROI
    roi = (net_profit / amount_usd) * 100 if amount_usd > 0 else 0
    
    return {
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "buy_fee": buy_fee,
        "sell_fee": sell_fee,
        "roi": roi,
        "buy_fee_rate": buy_fee_rate * 100,  # Convert to percentage
        "sell_fee_rate": sell_fee_rate * 100  # Convert to percentage
    }


def detect_arbitrage_opportunities(price_data: pd.DataFrame, min_profit_threshold: float = 1.0) -> List[Dict]:
    """Detect arbitrage opportunities from price data."""
    opportunities = []
    
    for asset, group in price_data.groupby("Asset"):
        valid_prices = group.dropna(subset=["Price"])
        if len(valid_prices) < 2:
            continue
            
        # Find best buy and sell prices
        best_buy = valid_prices.loc[valid_prices["Price"].idxmin()]
        best_sell = valid_prices.loc[valid_prices["Price"].idxmax()]
        
        if best_buy["Exchange"] == best_sell["Exchange"]:
            continue
            
        # Calculate profit
        profit_data = calculate_arbitrage_profit(
            best_buy["Price"], 
            best_sell["Price"], 
            best_buy["Exchange"], 
            best_sell["Exchange"],
            group.iloc[0]["Symbol"]  # Get symbol from the group
        )
        
        if profit_data["net_profit"] > min_profit_threshold:
            opportunities.append({
                "asset": asset,
                "buy_exchange": best_buy["Exchange"],
                "sell_exchange": best_sell["Exchange"],
                "buy_price": best_buy["Price"],
                "sell_price": best_sell["Price"],
                "spread": best_sell["Price"] - best_buy["Price"],
                "spread_pct": ((best_sell["Price"] - best_buy["Price"]) / best_buy["Price"]) * 100,
                **profit_data
            })
    
    return sorted(opportunities, key=lambda x: x["net_profit"], reverse=True)


with st.sidebar:
    st.header("Scanner Settings")
    selected_exchanges = st.multiselect(
        "Exchanges",
        list(SUPPORTED_EXCHANGES.keys()),
        default=list(SUPPORTED_EXCHANGES.keys()),
        format_func=lambda key: SUPPORTED_EXCHANGES[key],
    )

    selected_assets = st.multiselect(
        "Assets",
        list(SUPPORTED_ASSETS.keys()),
        default=list(DEFAULT_ASSETS),
    )

    timeframe_label = st.selectbox("Historical timeframe", list(TIMEFRAME_OPTIONS.keys()), index=2)
    timeframe = TIMEFRAME_OPTIONS[timeframe_label]

    history_days = st.slider("Historical window (days)", min_value=1, max_value=30, value=7)
    
    st.markdown("---")
    st.subheader("Live Monitoring")
    
    # Live monitoring controls
    auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=10)
    
    # Profit simulation settings
    st.markdown("---")
    st.subheader("Profit Simulation")
    simulation_amount = st.number_input("Simulation amount (USD)", min_value=100, max_value=100000, value=1000, step=100)
    min_profit_threshold = st.number_input("Minimum profit threshold (USD)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
    
    # Store threshold for use in arbitrage detection
    st.session_state.min_profit_threshold = min_profit_threshold

    st.markdown(
        """
        **Note:** Some exchanges may not be available in all regions due to regulatory restrictions.
        The app will automatically skip unavailable exchanges and continue with available ones.
        
        **Methodology Accuracy:**
        - Prices are fetched in real-time from exchange APIs
        - Transaction costs are estimated based on typical exchange fees
        - Profit calculations assume instant execution (theoretical)
        - Actual profits may vary due to slippage, network fees, and execution delays
        
        Need access to additional exchanges or private API data? Add your API credentials in
        Streamlit secrets. Public endpoints are used by default.
        """
    )

if not selected_exchanges:
    st.info("Please select at least one exchange to begin scanning.")
    st.stop()

if not selected_assets:
    st.info("Please select at least one asset to monitor.")
    st.stop()

# ---------------------------------------------------------------------------
# Auto-refresh functionality
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

# Real-time price comparison
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Real-time price snapshot")
with col2:
    if auto_refresh:
        st.caption(f"🔄 Auto-refreshing every {refresh_interval}s")

price_rows: List[Dict[str, Optional[float]]] = []
errors: List[str] = []
unavailable_exchanges: List[str] = []

# Check which exchanges are available
for exchange_id in selected_exchanges:
    try:
        get_exchange_client(exchange_id)
    except ccxt.ExchangeNotAvailable as e:
        unavailable_exchanges.append(SUPPORTED_EXCHANGES[exchange_id])
        errors.append(f"{SUPPORTED_EXCHANGES[exchange_id]}: {str(e)}")
    except Exception as exc:
        errors.append(f"{SUPPORTED_EXCHANGES[exchange_id]}: {exc}")

for asset in selected_assets:
    symbol = SUPPORTED_ASSETS[asset]
    for exchange_id in selected_exchanges:
        price = None
        try:
            price = fetch_latest_price(exchange_id, symbol)
        except Exception as exc:  # pragma: no cover - defensive logging for Streamlit UI
            errors.append(f"{SUPPORTED_EXCHANGES[exchange_id]} {asset}: {exc}")

        price_rows.append(
            {
                "Asset": asset,
                "Exchange": SUPPORTED_EXCHANGES[exchange_id],
                "Symbol": symbol,
                "Price": price,
            }
        )

if errors:
    with st.expander("⚠️ Exchange Warnings"):
        for message in errors:
            st.warning(message)
    
    if unavailable_exchanges:
        st.info(f"💡 **Tip:** Some exchanges ({', '.join(unavailable_exchanges)}) are not available in your region. The app will continue with available exchanges.")

price_table = pd.DataFrame(price_rows)
price_table_display = price_table.copy()
price_table_display["Price"] = price_table_display["Price"].map(lambda p: f"${p:,.2f}" if p else "N/A")
st.dataframe(price_table_display, hide_index=True, use_container_width=True)

# Enhanced arbitrage opportunities with profit calculation
threshold = getattr(st.session_state, 'min_profit_threshold', 1.0)
opportunities = detect_arbitrage_opportunities(price_table, threshold)

# Debug information
st.markdown("---")
st.subheader("🔍 Debug Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Threshold", f"${threshold:.2f}")
with col2:
    st.metric("Valid Price Pairs", len(price_table.dropna(subset=["Price"])))
with col3:
    st.metric("Opportunities Found", len(opportunities))

# Show current fees being used
st.markdown("#### 💰 Current Exchange Fees")
if not price_table.empty:
    sample_symbol = price_table.iloc[0]["Symbol"]
    fee_info = []
    for exchange_id in selected_exchanges:
        try:
            fees = get_exchange_fees(exchange_id, sample_symbol)
            fee_info.append({
                "Exchange": SUPPORTED_EXCHANGES[exchange_id],
                "Maker Fee": f"{fees['maker']*100:.3f}%",
                "Taker Fee": f"{fees['taker']*100:.3f}%",
                "Source": "Live API" if fees['maker'] != DEFAULT_TRANSACTION_COSTS.get(exchange_id, 0.1)/100 else "Default"
            })
        except:
            fee_info.append({
                "Exchange": SUPPORTED_EXCHANGES[exchange_id],
                "Maker Fee": "N/A",
                "Taker Fee": "N/A", 
                "Source": "Error"
            })
    
    if fee_info:
        fee_df = pd.DataFrame(fee_info)
        st.dataframe(fee_df, hide_index=True, use_container_width=True)

# Show all price differences for debugging
if not price_table.empty:
    st.markdown("#### Price Analysis")
    for asset, group in price_table.groupby("Asset"):
        valid_prices = group.dropna(subset=["Price"])
        if len(valid_prices) >= 2:
            min_price = valid_prices["Price"].min()
            max_price = valid_prices["Price"].max()
            spread = max_price - min_price
            spread_pct = (spread / min_price) * 100 if min_price > 0 else 0
            
            st.write(f"**{asset}:** Min: ${min_price:.2f}, Max: ${max_price:.2f}, Spread: ${spread:.2f} ({spread_pct:.2f}%)")
            
            # Show profit calculation for this asset
            best_buy = valid_prices.loc[valid_prices["Price"].idxmin()]
            best_sell = valid_prices.loc[valid_prices["Price"].idxmax()]
            
            if best_buy["Exchange"] != best_sell["Exchange"]:
                profit_data = calculate_arbitrage_profit(
                    best_buy["Price"], 
                    best_sell["Price"], 
                    best_buy["Exchange"], 
                    best_sell["Exchange"],
                    best_buy["Symbol"]
                )
                st.write(f"  → Net Profit: ${profit_data['net_profit']:.2f} (ROI: {profit_data['roi']:.2f}%)")
                st.write(f"  → Fees: {best_buy['Exchange']} {profit_data['buy_fee_rate']:.3f}%, {best_sell['Exchange']} {profit_data['sell_fee_rate']:.3f}%")

if opportunities:
    st.subheader("🚨 Live Arbitrage Opportunities")
    
    # Create enhanced display table
    opp_data = []
    for opp in opportunities:
        opp_data.append({
            "Asset": opp["asset"],
            "Buy Exchange": opp["buy_exchange"],
            "Sell Exchange": opp["sell_exchange"],
            "Buy Price": f"${opp['buy_price']:,.2f}",
            "Sell Price": f"${opp['sell_price']:,.2f}",
            "Spread": f"${opp['spread']:,.2f}",
            "Spread %": f"{opp['spread_pct']:.2f}%",
            "Net Profit": f"${opp['net_profit']:,.2f}",
            "ROI": f"{opp['roi']:.2f}%",
            "Buy Fee": f"${opp['buy_fee']:,.2f}",
            "Sell Fee": f"${opp['sell_fee']:,.2f}"
        })
    
    opp_df = pd.DataFrame(opp_data)
    st.dataframe(opp_df, hide_index=True, use_container_width=True)
    
    # Highlight best opportunities
    best_opportunity = opportunities[0]
    st.success(f"💰 **Best Opportunity:** {best_opportunity['asset']} - Net profit: ${best_opportunity['net_profit']:,.2f} ({best_opportunity['roi']:.2f}% ROI)")
    
    # Show profit breakdown for best opportunity
    with st.expander("📊 Profit Breakdown for Best Opportunity"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gross Profit", f"${best_opportunity['gross_profit']:,.2f}")
        with col2:
            st.metric("Total Fees", f"${best_opportunity['buy_fee'] + best_opportunity['sell_fee']:,.2f}")
        with col3:
            st.metric("Net Profit", f"${best_opportunity['net_profit']:,.2f}")
            
        st.markdown(f"**Strategy:** Buy {best_opportunity['asset']} on {best_opportunity['buy_exchange']} at ${best_opportunity['buy_price']:,.2f}, sell on {best_opportunity['sell_exchange']} at ${best_opportunity['sell_price']:,.2f}")
        
else:
    st.info("No arbitrage opportunities found above the minimum profit threshold. Try adjusting the threshold or monitoring more exchanges.")

# ---------------------------------------------------------------------------
# Historical analysis
# ---------------------------------------------------------------------------
st.subheader("Historical price comparison")
selected_asset_for_history = st.selectbox("Asset for historical chart", selected_assets, key="history_asset")
selected_symbol = SUPPORTED_ASSETS[selected_asset_for_history]

since_datetime = datetime.utcnow() - timedelta(days=history_days)
since_ms = int(since_datetime.timestamp() * 1000)

historical_frames: List[pd.DataFrame] = []
for exchange_id in selected_exchanges:
    frame = fetch_historical_prices(exchange_id, selected_symbol, timeframe, since_ms)
    if frame.empty:
        continue
    historical_frames.append(frame)

if historical_frames:
    combined_history = pd.concat(historical_frames).sort_values("timestamp")
    
    # Create subplots with secondary y-axis for spread
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Comparison Across Exchanges', 'Arbitrage Spread Analysis'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Color palette for exchanges
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Add price lines for each exchange
    for i, (exchange_name, exchange_df) in enumerate(combined_history.groupby("exchange")):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=exchange_df["timestamp"],
                y=exchange_df["close"],
                mode="lines+markers",
                name=exchange_name,
                line=dict(width=3, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=f"<b>{exchange_name}</b><br>" +
                             "Time: %{x}<br>" +
                             "Price: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )
    
    # Calculate and visualize arbitrage opportunities
    if len(historical_frames) >= 2:
        # Create pivot table for spread analysis
        pivot = combined_history.pivot_table(
            index="timestamp", columns="exchange", values="close", aggfunc="last"
        ).dropna()
        
        if not pivot.empty and len(pivot.columns) >= 2:
            # Calculate spreads at each timestamp
            pivot["max_price"] = pivot.max(axis=1)
            pivot["min_price"] = pivot.min(axis=1)
            pivot["spread"] = pivot["max_price"] - pivot["min_price"]
            pivot["spread_pct"] = (pivot["spread"] / pivot["min_price"]) * 100
            
            # Add spread line to second subplot
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot["spread_pct"],
                    mode="lines",
                    name="Spread %",
                    line=dict(width=2, color="purple"),
                    fill="tozeroy",
                    fillcolor="rgba(128, 0, 128, 0.1)",
                    hovertemplate="<b>Arbitrage Spread</b><br>" +
                                "Time: %{x}<br>" +
                                "Spread: %{y:.2f}%<br>" +
                                "Max Price: $" + pivot["max_price"].astype(str) + "<br>" +
                                "Min Price: $" + pivot["min_price"].astype(str) + "<extra></extra>",
                ),
                row=2, col=1
            )
            
            # Highlight significant spreads (>0.5%)
            significant_spreads = pivot[pivot["spread_pct"] > 0.5]
            
            if not significant_spreads.empty:
                # Add high spread markers on price chart
                fig.add_trace(
                    go.Scatter(
                        x=significant_spreads.index,
                        y=significant_spreads["max_price"],
                        mode="markers",
                        name="High Spread (Sell)",
                        marker=dict(
                            color="red",
                            size=12,
                            symbol="triangle-up",
                            line=dict(width=2, color="darkred")
                        ),
                        hovertemplate="<b>🚨 SELL OPPORTUNITY</b><br>" +
                                    "Time: %{x}<br>" +
                                    "Price: $%{y:.2f}<br>" +
                                    "Spread: %{customdata:.2f}%<br>" +
                                    "<i>Highest price - sell here</i><extra></extra>",
                        customdata=significant_spreads["spread_pct"],
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=significant_spreads.index,
                        y=significant_spreads["min_price"],
                        mode="markers",
                        name="High Spread (Buy)",
                        marker=dict(
                            color="green",
                            size=12,
                            symbol="triangle-down",
                            line=dict(width=2, color="darkgreen")
                        ),
                        hovertemplate="<b>💰 BUY OPPORTUNITY</b><br>" +
                                    "Time: %{x}<br>" +
                                    "Price: $%{y:.2f}<br>" +
                                    "Spread: %{customdata:.2f}%<br>" +
                                    "<i>Lowest price - buy here</i><extra></extra>",
                        customdata=significant_spreads["spread_pct"],
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add spread markers on spread chart
                fig.add_trace(
                    go.Scatter(
                        x=significant_spreads.index,
                        y=significant_spreads["spread_pct"],
                        mode="markers",
                        name="High Spread Events",
                        marker=dict(
                            color="orange",
                            size=8,
                            symbol="diamond",
                            line=dict(width=2, color="darkorange")
                        ),
                        hovertemplate="<b>🔥 High Spread Event</b><br>" +
                                    "Time: %{x}<br>" +
                                    "Spread: %{y:.2f}%<br>" +
                                    "Profit Potential: High<extra></extra>",
                        showlegend=True
                    ),
                    row=2, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=dict(
            text=f"📊 {selected_asset_for_history} Arbitrage Analysis - {history_days} Days",
            x=0.5,
            font=dict(size=20)
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text=f"Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=2, col=1)
    
    # Add profit threshold line
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="red",
        annotation_text="0.5% Threshold",
        annotation_position="top right",
        row=2, col=1
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.markdown("---")
    st.subheader("📈 Arbitrage Summary Statistics")
    
    pivot = combined_history.pivot_table(
        index="timestamp", columns="exchange", values="close", aggfunc="last"
    ).dropna()
    
    if not pivot.empty:
        # Calculate summary statistics
        pivot["max_price"] = pivot.max(axis=1)
        pivot["min_price"] = pivot.min(axis=1)
        pivot["spread"] = pivot["max_price"] - pivot["min_price"]
        pivot["spread_pct"] = (pivot["spread"] / pivot["min_price"]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Max Spread", 
                f"{pivot['spread_pct'].max():.2f}%",
                delta=f"${pivot['spread'].max():.2f}"
            )
        
        with col2:
            st.metric(
                "Avg Spread", 
                f"{pivot['spread_pct'].mean():.2f}%",
                delta=f"${pivot['spread'].mean():.2f}"
            )
        
        with col3:
            opportunities_count = len(pivot[pivot['spread_pct'] > 0.5])
            st.metric(
                "High Spread Events", 
                f"{opportunities_count}",
                delta=f"{opportunities_count/len(pivot)*100:.1f}% of time"
            )
        
        with col4:
            best_exchange = pivot.mean().idxmax()
            worst_exchange = pivot.mean().idxmin()
            st.metric(
                "Best Exchange", 
                best_exchange,
                delta=f"vs {worst_exchange}"
            )
        
        # Show detailed spread analysis
        with st.expander("📊 Detailed Spread Analysis"):
            st.markdown("#### Spread Distribution")
            spread_stats = pivot['spread_pct'].describe()
            st.dataframe(spread_stats.to_frame().round(3), use_container_width=True)
            
            st.markdown("#### Exchange Performance")
            exchange_performance = pivot.mean().sort_values(ascending=False)
            exchange_df = pd.DataFrame({
                'Exchange': exchange_performance.index,
                'Avg Price': exchange_performance.values,
                'Price Difference': exchange_performance.values - exchange_performance.min()
            })
            exchange_df['Price Difference %'] = (exchange_df['Price Difference'] / exchange_performance.min()) * 100
            st.dataframe(exchange_df.round(3), use_container_width=True, hide_index=True)
    
    else:
        st.info(
            "Historical spread analysis is unavailable because not all exchanges returned data"
            " for the selected timeframe."
        )
else:
    st.info(
        "Historical data could not be retrieved for the selected combination. Try reducing the"
        " look-back window or choosing a different timeframe."
    )

st.markdown(
    "---\n"
    "**Tip:** Use Streamlit's sidebar to add more assets. The app is built on public REST APIs"
    " via [ccxt](https://github.com/ccxt/ccxt). Provide API keys in Streamlit secrets for"
    " exchanges that require authentication."
)
