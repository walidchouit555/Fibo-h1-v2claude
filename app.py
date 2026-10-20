import requests
import streamlit as st
import pandas as pd
import numpy as np


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Gold Pattern Explorer",
    layout="wide"
)

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")
st.info(
    "يجلب التطبيق بيانات الذهب تلقائيًا عند الإمكان، مع الاحتفاظ بإمكانية "
    "رفع ملف CSV يدويًا كخيار بديل."
)

API_URL = "https://xaus.com/api/v1/chart"
MIN_HISTORY_MONTHS = 6
MAX_LATEST_CANDLE_AGE_HOURS = 72


# ==================================================
# 2. Data helpers
# ==================================================

def validate_history(dataframe):
    """Normalize OHLC history and reject invalid or insufficient data."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("لم تصل أي شموع صالحة من مصدر البيانات.")

    history = dataframe.copy()
    history.columns = [str(column).strip().lower() for column in history.columns]

    # Accept both CSV names and XAUS API's compact OHLCV names.
    history = history.rename(columns={
        "t": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(
            "الأعمدة المطلوبة غير موجودة: " + ", ".join(missing_columns)
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"], utc=True, errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "volume" in history.columns:
        history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    else:
        history["volume"] = np.nan

    history = history.dropna(subset=["timestamp"] + price_columns)
    history = history.sort_values("timestamp")
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    valid_ohlc = (
        (history["high"] >= history["low"])
        & (history["high"] >= history["open"])
        & (history["high"] >= history["close"])
        & (history["low"] <= history["open"])
        & (history["low"] <= history["close"])
        & (history[price_columns] > 0).all(axis=1)
    )
    history = history.loc[valid_ohlc].reset_index(drop=True)

    if history.empty:
        raise ValueError("لم تبقَ أي شموع صالحة بعد التحقق من الأسعار.")

    # Require at least six calendar months between the first and last candles.
    first_timestamp = history["timestamp"].min()
    last_timestamp = history["timestamp"].max()
    six_months_before_last = last_timestamp - pd.DateOffset(months=MIN_HISTORY_MONTHS)
    if first_timestamp > six_months_before_last:
        raise ValueError(
            "الفترة التاريخية المستلمة أقل من ستة أشهر. "
            f"الفترة المتاحة: {first_timestamp} — {last_timestamp}."
        )

    # The latest candle must not be excessively old. A 72-hour threshold allows
    # normal weekend/holiday gaps but rejects obviously stale automatic data.
    age = pd.Timestamp.now(tz="UTC") - last_timestamp
    if age > pd.Timedelta(hours=MAX_LATEST_CANDLE_AGE_HOURS):
        raise ValueError(
            "آخر شمعة قديمة جدًا للاستخدام التلقائي "
            f"(عمرها التقريبي {age.total_seconds() / 3600:.1f} ساعة)."
        )

    return history


def api_timestamp(value):
    """Parse XAUS timestamps expressed as Unix seconds/milliseconds or ISO text."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return pd.NaT
        unit = "ms" if abs(float(value)) >= 1e12 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_automatic_history():
    """Fetch two years of hourly XAU/USD candles and validate the response."""
    response = requests.get(
        API_URL,
        params={"symbol": "xau", "range": "2y", "interval": "1h"},
        timeout=25,
        headers={"User-Agent": "GoldPatternExplorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    data_state = payload.get("data_state", {}) if isinstance(payload, dict) else {}
    api_status = str(data_state.get("status", "")).lower() if isinstance(data_state, dict) else ""
    if api_status == "unavailable":
        raise ValueError("مصدر البيانات يعلن أن البيانات غير متاحة حاليًا.")
    if api_status == "stale":
        raise ValueError("مصدر البيانات يعلن أن البيانات التاريخية قديمة؛ لم تُستخدم تلقائيًا.")

    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = None
        for key in ("points", "data", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break
        if points is None:
            # Handle a standard chart-response wrapper if the upstream changes shape.
            chart = payload.get("chart")
            if isinstance(chart, dict):
                result = chart.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    timestamps = result[0].get("timestamp", [])
                    quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
                    points = [
                        {
                            "t": timestamps[i],
                            "o": (quote.get("open") or [None] * len(timestamps))[i],
                            "h": (quote.get("high") or [None] * len(timestamps))[i],
                            "l": (quote.get("low") or [None] * len(timestamps))[i],
                            "c": (quote.get("close") or [None] * len(timestamps))[i],
                            "v": (quote.get("volume") or [None] * len(timestamps))[i],
                        }
                        for i in range(len(timestamps))
                    ]
        if points is None:
            raise ValueError("صيغة استجابة مصدر البيانات غير معروفة.")
    else:
        raise ValueError("استجابة مصدر البيانات ليست بصيغة JSON المتوقعة.")

    if not points:
        raise ValueError("مصدر البيانات لم يُرجع أي شموع.")

    raw = pd.DataFrame(points)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    timestamp_column = next(
        (column for column in ("t", "timestamp", "time", "datetime", "date") if column in raw.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError("استجابة مصدر البيانات لا تحتوي على وقت الشمعة.")

    raw["timestamp"] = raw[timestamp_column].map(api_timestamp)
    raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if timestamp_column != "timestamp" and timestamp_column in raw.columns:
        # The parsed timestamp column above is authoritative.
        raw = raw.drop(columns=[timestamp_column])

    history = validate_history(raw)
    return history, data_state


def load_csv_history(uploaded_file):
    """Read an uploaded CSV using the same validation rules as automatic data."""
    csv_data = pd.read_csv(uploaded_file)
    return validate_history(csv_data)


# ==================================================
# 3. Automatic data fetching, with manual fallback
# ==================================================

if "auto_fetch_attempted" not in st.session_state:
    st.session_state["auto_fetch_attempted"] = True
    try:
        with st.spinner("جارٍ جلب بيانات الذهب التاريخية تلقائيًا..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
    except Exception as error:
        st.session_state["auto_history_data"] = None
        st.session_state["auto_data_state"] = {}
        st.session_state["auto_fetch_error"] = str(error)
        st.session_state["source_mode"] = "Upload CSV"

st.header("1. Data Source")

if st.button("🔄 Refresh automatic gold data"):
    try:
        fetch_automatic_history.clear()
        with st.spinner("جارٍ تحديث بيانات الذهب التاريخية..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
        st.success("تم تحديث البيانات التلقائية بنجاح.")
    except Exception as error:
        st.session_state["auto_fetch_error"] = str(error)
        st.warning(
            "تعذّر تحديث البيانات التلقائية. يمكنك متابعة العمل بملف CSV اليدوي. "
            f"التفاصيل: {error}"
        )

source_mode = st.radio(
    "Choose data source",
    options=["Automatic data", "Upload CSV"],
    horizontal=True,
    key="source_mode",
)

history = None

if source_mode == "Automatic data":
    auto_history = st.session_state.get("auto_history_data")
    if isinstance(auto_history, pd.DataFrame) and not auto_history.empty:
        history = auto_history
        st.success(
            f"تم تحميل {len(history):,} شمعة تاريخية تلقائيًا. "
            "المصدر: XAUS (XAU/USD، إطار الساعة)."
        )
        st.caption(
            "الفترة التاريخية (UTC): "
            f"{history['timestamp'].min()} — {history['timestamp'].max()}"
        )
        state = st.session_state.get("auto_data_state", {})
        if isinstance(state, dict) and state.get("as_of"):
            st.caption(f"حالة/وقت المصدر حسب API: {state.get('status', 'غير محدد')} — {state['as_of']}")
        st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
    else:
        error_message = st.session_state.get("auto_fetch_error")
        st.warning(
            "البيانات التلقائية غير متاحة حاليًا. اختر «Upload CSV» "
            "للعودة إلى الطريقة القديمة، أو اضغط زر التحديث لإعادة المحاولة."
        )
        if error_message:
            st.caption(f"تفاصيل فشل الجلب التلقائي: {error_message}")
else:
    history_file = st.file_uploader(
        "Upload historical gold data (CSV)",
        type=["csv"],
        key="history",
        help="يجب أن يحتوي الملف على timestamp, open, high, low, close."
    )
    if history_file is not None:
        try:
            history = load_csv_history(history_file)
            st.session_state["csv_history_data"] = history
            st.success(f"Loaded {len(history):,} valid candles from CSV.")
            st.caption(
                "الفترة التاريخية (UTC): "
                f"{history['timestamp'].min()} — {history['timestamp'].max()}"
            )
            st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"تعذرت قراءة ملف CSV أو التحقق منه: {error}")
    elif isinstance(st.session_state.get("csv_history_data"), pd.DataFrame):
        history = st.session_state["csv_history_data"].copy()
        st.info("تم الاحتفاظ بملف CSV الذي رفعته سابقًا خلال هذه الجلسة.")

if isinstance(history, pd.DataFrame):
    st.session_state["history_data"] = history
else:
    st.session_state.pop("history_data", None)



# ==================================================
# 4. Backtest engine (cached, plain Python lists for speed)
# ==================================================

@st.cache_data(show_spinner=False)
def run_backtest(frame, entry_fib, target_extension, max_wait, pending_mode):
    """Walk-forward simulation. `frame` has timestamp/open/high/low/close/signal."""
    ts = frame["timestamp"].tolist()
    opens = frame["open"].tolist()
    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    closes = frame["close"].tolist()
    signals = frame["signal"].tolist()
    n = len(frame)
    open_label = "Open at end of data (marked to market)"
    keep_newest = pending_mode == "Keep only the newest signal pending"
    ext = target_extension / 100.0
    fib = entry_fib / 100.0

    trades = []
    pending = None
    trade = None
    replaced = invalidated = expired = 0

    def make_signal(i):
        high, low = highs[i], lows[i]
        span = high - low
        if span <= 0:
            return None
        bullish = closes[i] >= opens[i]
        if bullish:
            entry, stop, target = high - span * fib, low, low + span * ext
        else:
            entry, stop, target = low + span * fib, high, high - span * ext
        return {
            "signal_index": i, "signal_time": ts[i],
            "direction": "Buy" if bullish else "Sell", "bullish": bullish,
            "entry": entry, "initial_stop": stop, "target": target,
        }

    def finish(t, exit_index, exit_price, reason):
        reward = (exit_price - t["entry"]) if t["bullish"] else (t["entry"] - exit_price)
        risk = abs(t["entry"] - t["initial_stop"])
        trades.append({
            "Signal time (UTC)": t["signal_time"],
            "Entry time (UTC)": ts[t["entry_index"]],
            "Exit time (UTC)": ts[exit_index],
            "Direction": t["direction"],
            "Entry": t["entry"],
            "Initial stop": t["initial_stop"],
            "Stop at exit": t["current_stop"],
            f"Target {target_extension:g}% extension": t["target"],
            "Exit price": exit_price,
            "Exit reason": reason,
            "Result (R)": reward / risk if risk > 0 else float("nan"),
            "Result (%) of entry price": (reward / t["entry"] * 100.0) if t["entry"] else float("nan"),
        })

    for i in range(n):
        o, h, l = opens[i], highs[i], lows[i]
        had_open = trade is not None

        # 1) Manage a trade opened before this candle.
        if trade is not None and i > trade["entry_index"]:
            stop = trade["current_stop"]
            if trade["bullish"]:
                hit_stop, hit_target = l <= stop, h >= trade["target"]
            else:
                hit_stop, hit_target = h >= stop, l <= trade["target"]
            if hit_stop:
                if trade["bullish"] and o < stop:
                    fill = o
                elif (not trade["bullish"]) and o > stop:
                    fill = o
                else:
                    fill = stop
                finish(trade, i, fill, trade["current_stop_reason"])
                trade = None
            elif hit_target:
                finish(trade, i, trade["target"], f"Take profit {target_extension:g}% extension")
                trade = None

        # 2) Validate the pending signal.
        if pending is not None and i > pending["signal_index"]:
            if max_wait > 0 and i - pending["signal_index"] > max_wait:
                pending = None
                expired += 1
            elif (pending["bullish"] and l <= pending["initial_stop"]) or (
                (not pending["bullish"]) and h >= pending["initial_stop"]
            ):
                pending = None
                invalidated += 1

        # 3) Entry only if no position was open at the start of this candle.
        if not had_open and trade is None and pending is not None and i > pending["signal_index"]:
            stop_crossed = (pending["bullish"] and l <= pending["initial_stop"]) or (
                (not pending["bullish"]) and h >= pending["initial_stop"]
            )
            if stop_crossed:
                pending = None
                invalidated += 1
            elif l <= pending["entry"] <= h:
                trade = dict(pending)
                trade["entry_index"] = i
                trade["current_stop"] = trade["initial_stop"]
                trade["current_stop_reason"] = "Initial stop loss"
                pending = None

        # 4) New signal (confirmed at close, cannot enter on the same candle).
        if signals[i]:
            new_signal = make_signal(i)
            if new_signal is not None:
                if trade is not None:
                    if keep_newest:
                        if pending is not None:
                            replaced += 1
                        pending = new_signal
                else:
                    if pending is not None:
                        replaced += 1
                    pending = new_signal

    if trade is not None:
        finish(trade, n - 1, closes[n - 1], open_label)

    return pd.DataFrame(trades), pending, replaced, invalidated, expired


# ==================================================
# 5. Volatility Breakout Strategy + historical backtest
# ==================================================

st.divider()
st.header("2. Volatility Breakout Strategy — configurable Fibonacci")
st.write(
    "تبحث الاستراتيجية عن شمعة يتجاوز مداها (High − Low) أكبر مدى في الشموع "
    "السابقة، ثم تنتظر رجوع السعر إلى مستوى تصحيح فيبوناتشي الذي تختاره، "
    "وتغلق عند امتداد فيبوناتشي الذي تحدده كهدف."
)

with st.expander("Strategy settings", expanded=True):
    lookback_hours = st.number_input(
        "Reference window: previous hourly candles",
        min_value=3, max_value=144, value=24, step=1,
        help="تُقارن كل شمعة بالشموع السابقة فقط، ولا تدخل ضمن نافذتها المرجعية."
    )
    fib_retracement_levels = [23.6, 38.2, 50.0, 61.8, 78.6]
    entry_fib = st.selectbox(
        "Entry — Fibonacci Retracement",
        options=fib_retracement_levels,
        index=0,
        format_func=lambda value: f"{value:g}%",
        help="اختر مستوى التصحيح الذي ينتظر التطبيق وصول السعر إليه للدخول."
    )
    fib_extension_levels = [100.0, 127.2, 141.0, 161.8, 200.0, 261.8]
    target_extension = st.selectbox(
        "Final target — Fibonacci Extension",
        options=fib_extension_levels,
        index=3,
        format_func=lambda value: f"{value:g}%",
        help="اختر امتداد فيبوناتشي الذي سيكون هدف إغلاق الصفقة."
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Entry retracement", f"{entry_fib:g}%")
    col_b.metric("Final target extension", f"{target_extension:g}%")
    max_wait = st.number_input(
        "Maximum waiting candles after signal (0 = unlimited)",
        min_value=0, max_value=50000, value=0, step=1,
        help=(
            "الصفر يعني الانتظار دون حد زمني. أي إشارة أحدث تستبدل الإشارة "
            "المنتظرة القديمة. لا يمكن استخدام الإشارة نفسها لفتح أكثر من صفقة."
        )
    )
    pending_mode = st.selectbox(
        "New signals while a trade is open",
        options=[
            "Keep only the newest signal pending",
            "Ignore signals while a trade is open",
        ],
        index=0,
        help=(
            "الخيار الافتراضي يحتفظ بآخر إشارة فقط حتى تُغلق الصفقة الحالية؛ "
            "ولا يسمح بفتح صفقتين متزامنتين."
        )
    )
    st.subheader("Demo account")
    capital_col, risk_col = st.columns(2)
    initial_capital = capital_col.number_input(
        "Initial demo capital (USD)",
        min_value=1.0, max_value=100000000.0, value=1000.0, step=100.0,
        help="رأس المال الافتراضي الذي يبدأ به الاختبار، ويمكنك تغييره."
    )
    risk_percent = risk_col.number_input(
        "Risk per trade (%)",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="النسبة من رصيد الحساب الحالية التي تمثل مخاطرة 1R في كل صفقة. الافتراضي 1%."
    )
    st.caption(
        "حساب تجريبي افتراضي: تُحسب كل صفقة على أساس نتيجة R ومخاطرة ثابتة كنسبة من "
        "رصيد الحساب الحالي. لا يشمل السبريد أو العمولة أو الانزلاق، وليس حسابًا حقيقيًا."
    )

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات في الأعلى أولًا.")
else:
    strat_history = st.session_state["history_data"].reset_index(drop=True)
    strat_history = strat_history.assign(
        range=strat_history["high"] - strat_history["low"]
    )
    strat_history["previous_max_range"] = (
        strat_history["range"].shift(1).rolling(int(lookback_hours)).max()
    )
    strat_history["signal"] = (
        strat_history["previous_max_range"].notna()
        & (strat_history["range"] > strat_history["previous_max_range"])
        & (strat_history["range"] > 0)
    )

    signal_rows = strat_history[strat_history["signal"]]
    if len(strat_history) <= int(lookback_hours) + 2:
        st.warning("لا توجد شموع كافية لتشغيل الاستراتيجية.")
    else:
        st.caption(
            f"عدد الشموع: {len(strat_history):,} | "
            f"الفترة: {strat_history['timestamp'].iloc[0]} — "
            f"{strat_history['timestamp'].iloc[-1]}"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Exceptional candles", f"{len(signal_rows):,}")
        col2.metric("Reference hours", int(lookback_hours))
        col3.metric("Entry / Final target", f"{entry_fib:g}% / {target_extension:g}%")

        if not signal_rows.empty:
            display_signals = signal_rows[
                ["timestamp", "open", "high", "low", "close", "range", "previous_max_range"]
            ].tail(200).copy()
            display_signals["range"] = display_signals["range"].round(5)
            display_signals["previous_max_range"] = display_signals["previous_max_range"].round(5)
            st.subheader("Detected exceptional candles")
            st.dataframe(display_signals, use_container_width=True, hide_index=True)
            st.caption("يعرض الجدول أحدث 200 إشارة كحد أقصى.")

        st.subheader("Walk-forward backtest")
        st.caption(
            "قواعد الاختبار: لا دخول في شمعة الإشارة نفسها؛ وقف الخسارة ثابت عند الطرف "
            "المقابل لشمعة الإشارة؛ إذا لُمِس الوقف والهدف داخل شمعة واحدة يُحتسب الوقف أولًا. "
            "لا توجد قاعدة 68% ولا تتبّع لوقف الخسارة. النتائج لا تشمل السبريد أو العمولة أو الانزلاق."
        )

        trade_df, pending_signal, replaced_signals, invalidated_signals, expired_signals = run_backtest(
            strat_history[["timestamp", "open", "high", "low", "close", "signal"]],
            float(entry_fib), float(target_extension), int(max_wait), pending_mode,
        )

        open_label = "Open at end of data (marked to market)"
        closed = pd.DataFrame()
        has_open_end = False
        if not trade_df.empty:
            is_open_end = trade_df["Exit reason"] == open_label
            has_open_end = bool(is_open_end.any())
            closed = trade_df[~is_open_end]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Closed trades", f"{len(closed):,}")
        col_b.metric("Open at end", "1" if has_open_end else "0")
        col_c.metric("Signals replaced", f"{replaced_signals:,}")

        if trade_df.empty:
            st.info("لم تُنتج البيانات صفقات دخول وفق الشروط الحالية.")
        else:
            results_r = closed["Result (R)"] if not closed.empty else pd.Series(dtype=float)
            wins = int((results_r > 0).sum())
            losses = int((results_r < 0).sum())
            breakeven = int((results_r.abs() <= 1e-10).sum())
            total_closed = len(closed)
            win_rate = wins / total_closed * 100.0 if total_closed else 0.0
            total_r = float(results_r.sum()) if total_closed else 0.0
            avg_r = float(results_r.mean()) if total_closed else 0.0
            gross_profit = float(results_r[results_r > 0].sum()) if total_closed else 0.0
            gross_loss = float(-results_r[results_r < 0].sum()) if total_closed else 0.0
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0
                else (float("inf") if gross_profit > 0 else float("nan"))
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win rate (closed trades)", f"{win_rate:.2f}%")
            m2.metric("Net result", f"{total_r:.2f} R")
            m3.metric("Average result", f"{avg_r:.2f} R")
            m4.metric(
                "Profit factor",
                "∞" if np.isinf(profit_factor) else (
                    "—" if pd.isna(profit_factor) else f"{profit_factor:.2f}"
                )
            )

            # Simulated account: 1R risks the configured percentage of current equity.
            demo_equity = float(initial_capital)
            risk_fraction = float(risk_percent) / 100.0
            equity_rows = [(0, strat_history["timestamp"].iloc[0], 0.0, demo_equity, 0.0, demo_equity)]
            if not closed.empty:
                ordered = closed.sort_values("Exit time (UTC)")
                for number, (exit_time, trade_r) in enumerate(
                    zip(ordered["Exit time (UTC)"], ordered["Result (R)"].astype(float)), start=1
                ):
                    before = demo_equity
                    demo_equity = max(0.0, before + before * risk_fraction * trade_r)
                    equity_rows.append((number, exit_time, trade_r, before, demo_equity - before, demo_equity))
            equity_df = pd.DataFrame(
                equity_rows,
                columns=["Trade", "Exit time (UTC)", "Result (R)", "Balance before (USD)",
                         "Profit/Loss (USD)", "Balance after (USD)"],
            )
            net_usd = demo_equity - float(initial_capital)
            return_pct = (net_usd / float(initial_capital)) * 100.0 if initial_capital else 0.0
            st.subheader("$1,000 demo account results")
            cap1, cap2, cap3, cap4 = st.columns(4)
            cap1.metric("Initial capital", f"${initial_capital:,.2f}")
            cap2.metric("Final capital", f"${demo_equity:,.2f}")
            cap3.metric("Net profit / loss", f"${net_usd:,.2f}", delta=f"{return_pct:+.2f}%")
            cap4.metric("Risk per trade", f"{risk_percent:g}%")
            st.caption(
                f"تم احتساب {len(closed)} صفقة مغلقة فقط، مع إعادة استثمار الرصيد بعد كل صفقة. "
                "الصفقات المفتوحة عند نهاية البيانات لا تدخل في رأس المال النهائي."
            )
            st.line_chart(equity_df.set_index("Trade")["Balance after (USD)"])
            st.download_button(
                "Download demo account balance (CSV)",
                data=equity_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="gold_pattern_demo_account.csv",
                mime="text/csv",
            )
            st.dataframe(trade_df.round(5), use_container_width=True, hide_index=True)
            st.download_button(
                "Download backtest trades (CSV)",
                data=trade_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="volatility_breakout_backtest_v2.csv",
                mime="text/csv",
            )
            if total_closed:
                st.write(
                    f"الرابحة: {wins} | الخاسرة: {losses} | "
                    f"التعادل: {breakeven} | متوسط النتيجة: {avg_r:.2f}R"
                )

        if pending_signal is not None:
            waited = (len(strat_history) - 1) - pending_signal["signal_index"]
            st.info(
                f"هناك إشارة أخيرة ما زالت تنتظر الدخول: {pending_signal['direction']}، "
                f"من {pending_signal['signal_time']}، مستوى الدخول {pending_signal['entry']:.5f}. "
                f"عدد الشموع المنقضية منذ الإشارة: {waited}."
            )
        st.caption(
            f"إشارات أُلغيت بسبب تجاوز الوقف الأصلي: {invalidated_signals:,} | "
            f"إشارات انتهى وقت انتظارها: {expired_signals:,}. "
            "إذا كان الانتظار = 0، فلا تنتهي الإشارة بسبب الوقت؛ لكن إشارة أحدث قد تستبدلها."
        )
        st.warning(
            "هذا اختبار تاريخي افتراضي، وليس ضمانًا للربح أو توصية بالتداول. "
            "الشموع ذات إطار الساعة لا تكشف ترتيب السعر داخل الشمعة؛ لذلك استُخدمت "
            "افتراضات محافظة، ولم تُحتسب تكاليف التداول."
        )
