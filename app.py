import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(
    page_title="Cyber Attack Predictor",
    page_icon="🫧",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(0, 255, 170, 0.08), transparent 35%),
                radial-gradient(circle at 85% 20%, rgba(0, 221, 255, 0.10), transparent 35%),
                linear-gradient(145deg, #061018 0%, #08141f 45%, #0b1928 100%);
            color: #d8f6ff;
        }

        .bubble-wrap {
            position: fixed;
            inset: 0;
            overflow: hidden;
            z-index: -1;
            pointer-events: none;
        }

        .bubble {
            position: absolute;
            border-radius: 50%;
            opacity: 0.12;
            filter: blur(0.5px);
            animation: floatUp linear infinite;
        }

        .bubble.b1 { width: 220px; height: 220px; left: 8%;  bottom: -260px; background: #00e1ff; animation-duration: 26s; }
        .bubble.b2 { width: 120px; height: 120px; left: 24%; bottom: -150px; background: #00ff9f; animation-duration: 22s; animation-delay: 3s; }
        .bubble.b3 { width: 180px; height: 180px; left: 45%; bottom: -210px; background: #15a7ff; animation-duration: 28s; animation-delay: 1s; }
        .bubble.b4 { width: 90px;  height: 90px;  left: 62%; bottom: -120px; background: #17ffd6; animation-duration: 20s; animation-delay: 5s; }
        .bubble.b5 { width: 200px; height: 200px; left: 80%; bottom: -240px; background: #3ac6ff; animation-duration: 30s; animation-delay: 2s; }

        @keyframes floatUp {
            0%   { transform: translateY(0) translateX(0); }
            50%  { transform: translateY(-55vh) translateX(18px); }
            100% { transform: translateY(-115vh) translateX(-16px); }
        }

        .soft-card {
            background: rgba(8, 25, 39, 0.72);
            border: 1px solid rgba(27, 222, 199, 0.35);
            border-radius: 18px;
            padding: 16px 18px;
            backdrop-filter: blur(3px);
            box-shadow: 0 10px 28px rgba(0, 255, 170, 0.10);
            margin-bottom: 0.75rem;
        }

        .title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #9efcff;
            margin-bottom: 0.2rem;
            letter-spacing: 0.5px;
        }

        .subtitle {
            color: #8fc1d6;
            margin-bottom: 0.9rem;
            font-size: 1rem;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(0, 255, 170, 0.4);
            color: #9efcdf;
            background: rgba(0, 255, 170, 0.08);
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }

        .section-title {
            color: #9df4ff;
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        [data-testid="stTabs"] {
            background: rgba(6, 18, 29, 0.45);
            border: 1px solid rgba(24, 196, 240, 0.2);
            border-radius: 14px;
            padding: 10px;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMarkdownContainer"] p,
        label {
            color: #d8f6ff !important;
        }

        .stButton > button {
            background: linear-gradient(90deg, #00c2ff 0%, #00ff9f 100%);
            color: #04131f;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.52rem 1rem;
        }

        .stButton > button:hover {
            filter: brightness(1.06);
            transform: translateY(-1px);
        }

        .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 170, 0.45);
            background: rgba(0, 255, 170, 0.10);
            color: #bafee6;
        }

        .stDataFrame {
            border: 1px solid rgba(24, 196, 240, 0.25);
            border-radius: 12px;
            overflow: hidden;
        }
        </style>

        <div class="bubble-wrap">
            <div class="bubble b1"></div>
            <div class="bubble b2"></div>
            <div class="bubble b3"></div>
            <div class="bubble b4"></div>
            <div class="bubble b5"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_bundle() -> dict:
    candidates = [
        Path("cyber_attack_model.pkl"),
        Path("..").joinpath("cyber_attack_model.pkl"),
    ]
    for path in candidates:
        if path.exists():
            with path.open("rb") as f:
                bundle = pickle.load(f)
            return bundle
    raise FileNotFoundError(
        "Could not find cyber_attack_model.pkl in current or parent directory."
    )


def safe_ip_octets(ip_value: str) -> list[int]:
    try:
        parts = str(ip_value).split(".")
        values = []
        for token in parts[:4]:
            if token.isdigit():
                values.append(int(token))
            else:
                values.append(0)
        while len(values) < 4:
            values.append(0)
        return values
    except Exception:
        return [0, 0, 0, 0]


def browser_family(value: str) -> str:
    s = str(value).lower()
    if "chrome" in s:
        return "Chrome"
    if "firefox" in s:
        return "Firefox"
    if "msie" in s or "trident" in s:
        return "IE"
    if "safari" in s and "chrome" not in s:
        return "Safari"
    if "opera" in s:
        return "Opera"
    if "mozilla" in s:
        return "Mozilla"
    return "Other"


def os_family(value: str) -> str:
    s = str(value).lower()
    if "windows" in s:
        return "Windows"
    if "mac os x" in s or "macintosh" in s:
        return "Mac"
    if "android" in s:
        return "Android"
    if "iphone" in s:
        return "iPhone"
    if "ipad" in s:
        return "iPad"
    if "linux" in s:
        return "Linux"
    if "chromeos" in s:
        return "ChromeOS"
    return "Other"


def os_version(value: str) -> str:
    patterns = [
        r"Windows NT ([0-9\.]+)",
        r"Mac OS X ([0-9_\.]+)",
        r"Android ([0-9\.]+)",
        r"iPhone OS ([0-9_]+)",
        r"iPad.*OS ([0-9_]+)",
        r"Linux",
        r"ChromeOS",
    ]
    text = str(value)
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue

        if m.lastindex and m.group(1) is not None:
            return str(m.group(1)).replace("_", ".")

        # Patterns like 'Linux' / 'ChromeOS' have no capture group
        return m.group(0)
    return "Unknown"


def ensure_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "Timestamp": "2024-01-01 12:00:00",
        "Source IP Address": "192.168.1.10",
        "Destination IP Address": "8.8.8.8",
        "Source Port": 50000,
        "Destination Port": 443,
        "Protocol": "TCP",
        "Packet Length": 900,
        "Packet Type": "Data",
        "Traffic Type": "HTTP",
        "Payload Data": "normal_payload",
        "Malware Indicators": "",
        "Severity Level": "Medium",
        "Device Information": "Mozilla/5.0 (Windows NT 10.0)",
        "IDS/IPS Alerts": np.nan,
    }
    for key, val in defaults.items():
        if key not in raw_df.columns:
            raw_df[key] = val
    return raw_df


def build_feature_matrix(raw_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    data = ensure_columns(raw_df.copy())

    for col in ["Source Port", "Destination Port", "Packet Length"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data["hour"] = data["Timestamp"].dt.hour.fillna(12)
    data["dayofweek"] = data["Timestamp"].dt.dayofweek.fillna(0)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["off_hours"] = data["hour"].between(0, 5).astype(int)

    data["high_src_port"] = (data["Source Port"] > 1024).fillna(0).astype(int)
    data["high_dst_port"] = (data["Destination Port"] > 1024).fillna(0).astype(int)
    data["same_port"] = (data["Source Port"] == data["Destination Port"]).fillna(0).astype(int)
    data["dst_is_web_port"] = data["Destination Port"].isin([80, 443, 8080, 53]).fillna(False).astype(int)

    data["payload_len"] = data["Payload Data"].astype(str).str.len()
    data["payload_has_ioc"] = (
        data["Malware Indicators"].astype(str).str.contains("IoC", na=False).astype(int)
    )

    packet_len = data["Packet Length"].fillna(data["Packet Length"].median())
    packet_len = packet_len.fillna(0)
    data["PacketLength_log"] = np.log1p(packet_len.clip(lower=0))
    data["payload_to_packet_ratio"] = data["payload_len"] / (packet_len + 1.0)

    src_octets = np.array([safe_ip_octets(v) for v in data["Source IP Address"]])
    dst_octets = np.array([safe_ip_octets(v) for v in data["Destination IP Address"]])

    for i in range(4):
        data[f"src_octet{i+1}"] = src_octets[:, i]
        data[f"dst_octet{i+1}"] = dst_octets[:, i]

    ua = data["Device Information"].fillna("").astype(str)
    data["Browser"] = ua.str.split("/").str[0].str.strip().replace("", "Unknown")
    data["BrowserMajor"] = ua.str.extract(r"/([0-9]+)")[0].fillna("Unknown")
    data["BrowserFamily"] = ua.apply(browser_family)
    data["OSFamily"] = ua.apply(os_family)
    data["OSVersion"] = ua.apply(os_version)
    data["IsMobile"] = ua.str.contains(r"android|iphone|ipod", case=False, regex=True).astype(int)
    data["IsTablet"] = ua.str.contains(r"ipad", case=False, regex=True).astype(int)
    data["IsDesktop"] = ((data["IsMobile"] == 0) & (data["IsTablet"] == 0)).astype(int)
    data["UA_Length"] = ua.str.len()
    data["UA_TokenCount"] = ua.str.split().apply(len)

    cat_cols = [
        "Packet Type",
        "Traffic Type",
        "Severity Level",
        "Browser",
        "BrowserMajor",
        "BrowserFamily",
        "OSFamily",
        "OSVersion",
    ]

    X = data.copy()
    for cat in cat_cols:
        dummies = pd.get_dummies(X[cat].astype(str), prefix=cat, drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def predict_with_model(input_df: pd.DataFrame, model, class_names: list[str], feature_columns: list[str]) -> pd.DataFrame:
    if set(feature_columns).issubset(set(input_df.columns)):
        X = input_df[feature_columns].copy()
    else:
        X = build_feature_matrix(input_df, feature_columns)

    proba = model.predict(X)
    if isinstance(proba, list):
        proba = np.array(proba)
    if proba.ndim == 1:
        proba = np.vstack([1 - proba, proba]).T

    pred_idx = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    result = input_df.copy().reset_index(drop=True)
    result["Predicted Class"] = [class_names[i] for i in pred_idx]
    result["Confidence"] = np.round(confidence, 4)

    for i, name in enumerate(class_names):
        if i < proba.shape[1]:
            result[f"P({name})"] = np.round(proba[:, i], 4)

    return result


def map_attack_type_to_3class(label: str) -> float:
    s = str(label).strip().lower()
    if "ddos" in s:
        return 0
    if any(k in s for k in ["intrusion", "bruteforce", "brute force", "sql injection", "xss", "probe", "scan"]):
        return 1
    if any(k in s for k in ["malware", "ransomware", "trojan", "worm", "botnet", "spyware"]):
        return 2
    return np.nan


def generate_target_from_raw(raw_df: pd.DataFrame) -> pd.Series:
    data = ensure_columns(raw_df.copy())

    for col in ["Source Port", "Destination Port", "Packet Length"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data["hour"] = data["Timestamp"].dt.hour.fillna(12)
    data["high_src_port"] = (data["Source Port"] > 1024).fillna(0).astype(int)
    data["high_dst_port"] = (data["Destination Port"] > 1024).fillna(0).astype(int)
    data["same_port"] = (data["Source Port"] == data["Destination Port"]).fillna(0).astype(int)
    data["dst_is_web_port"] = data["Destination Port"].isin([80, 443, 8080, 53]).fillna(False).astype(int)
    data["payload_len"] = data["Payload Data"].astype(str).str.len()
    data["payload_has_ioc"] = data["Malware Indicators"].astype(str).str.contains("IoC", na=False).astype(int)

    packet_len = data["Packet Length"].fillna(data["Packet Length"].median()).fillna(0)
    data["PacketLength_log"] = np.log1p(packet_len.clip(lower=0))
    data["payload_to_packet_ratio"] = data["payload_len"] / (packet_len + 1.0)
    proto_upper = data["Protocol"].astype(str).str.upper()
    data["is_tcp"] = (proto_upper == "TCP").astype(int)
    data["is_udp"] = (proto_upper == "UDP").astype(int)
    data["is_icmp"] = (proto_upper == "ICMP").astype(int)

    pkt_q75 = data["PacketLength_log"].quantile(0.75)
    pkt_q50 = data["PacketLength_log"].quantile(0.50)
    ratio_q75 = data["payload_to_packet_ratio"].quantile(0.75)
    payload_q60 = data["payload_len"].quantile(0.60)

    def score_row(row: pd.Series) -> np.ndarray:
        protocol = str(row.get("Protocol", "")).upper()
        traffic_type = str(row.get("Traffic Type", "")).lower()
        ids_alert = pd.notna(row.get("IDS/IPS Alerts", None))

        score_ddos = 0.0
        score_intrusion = 0.0
        score_malware = 0.0

        score_ddos += 2.2 * row.get("is_tcp", 0)
        score_ddos += 1.8 * row.get("dst_is_web_port", 0)
        score_ddos += 1.2 * int(row.get("hour", 12) <= 5)
        score_ddos += 1.4 * (row.get("PacketLength_log", 0) >= pkt_q75)
        score_ddos += 0.7 * (row.get("PacketLength_log", 0) >= pkt_q50)

        score_intrusion += 2.0 * row.get("high_src_port", 0)
        score_intrusion += 1.4 * row.get("high_dst_port", 0)
        score_intrusion += 1.2 * row.get("is_udp", 0)
        score_intrusion += 1.0 * ids_alert
        score_intrusion += 0.8 * row.get("same_port", 0)

        score_malware += 2.4 * row.get("payload_has_ioc", 0)
        score_malware += 1.5 * row.get("is_icmp", 0)
        score_malware += 1.2 * (row.get("payload_to_packet_ratio", 0) >= ratio_q75)
        score_malware += 1.0 * (row.get("payload_len", 0) >= payload_q60)
        score_malware += 1.0 * (traffic_type in ["malicious", "suspicious"])

        if row.get("payload_has_ioc", 0) == 1:
            score_ddos -= 0.8
        if row.get("dst_is_web_port", 0) == 1 and row.get("is_tcp", 0) == 1:
            score_malware -= 0.6
        if protocol == "ICMP":
            score_intrusion -= 0.4
        if ids_alert:
            score_ddos -= 0.4

        return np.array([max(score_ddos, 0), max(score_intrusion, 0), max(score_malware, 0)])

    score_matrix = np.vstack(data.apply(score_row, axis=1).values)
    return pd.Series(np.argmax(score_matrix, axis=1), index=raw_df.index)


def index_to_class(labels_idx: pd.Series, class_names: list[str]) -> pd.Series:
    return labels_idx.map(lambda x: class_names[int(x)] if pd.notna(x) else "Unknown")


def render_distribution_and_confidence(pred_df: pd.DataFrame, generated_labels: pd.Series) -> None:
    st.markdown("### Distribution & Confidence")

    dist_df = pd.DataFrame(
        {
            "Predicted": pred_df["Predicted Class"].value_counts(),
            "Generated": generated_labels.value_counts(),
        }
    ).fillna(0)

    st.dataframe(dist_df.astype(int), use_container_width=True)
    st.bar_chart(dist_df)

    st.markdown("**Confidence summary**")
    conf = pred_df["Confidence"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Confidence", f"{conf.mean():.4f}")
    m2.metric("Min Confidence", f"{conf.min():.4f}")
    m3.metric("Low Confidence Rate (<0.60)", f"{(conf < 0.60).mean():.2%}")

    conf_bins = pd.cut(conf, bins=[0, 0.5, 0.7, 0.85, 1.0], include_lowest=True)
    st.dataframe(conf_bins.value_counts().sort_index().rename("Rows").to_frame(), use_container_width=True)


def main() -> None:
    inject_styles()

    st.markdown('<div class="title">🛡️ Cyber Threat Command Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time threat class inference from network telemetry using your deployed ML model.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge">Threat Detection</span><span class="badge">Batch Scoring</span><span class="badge">Feature-Aligned Inference</span>',
        unsafe_allow_html=True,
    )

    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        st.stop()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    class_names = bundle.get("class_names", ["DDoS", "Intrusion", "Malware"])

    with st.container(border=False):
        st.markdown(
            '<div class="soft-card"><div class="section-title">Model Status</div>'
            'Model ready: <b>cyber_attack_model.pkl</b><br>'
            f'Input features expected: <b>{len(feature_columns)}</b><br>'
            f'Output classes: <b>{", ".join(class_names)}</b></div>',
            unsafe_allow_html=True,
        )

    tab_manual, tab_csv = st.tabs(["Manual Input", "CSV Upload"])

    with tab_manual:
        st.subheader("Single Prediction")
        st.caption("Use this for quick triage of one network event.")
        presets = {
            "ddos": {
                "source_port": 54021,
                "destination_port": 443,
                "packet_length": 1480,
                "protocol": "TCP",
                "packet_type": "Data",
                "traffic_type": "HTTP",
                "severity": "High",
                "timestamp": "2024-01-07 02:14:22",
                "src_ip": "185.23.14.201",
                "dst_ip": "104.26.10.78",
                "payload": "flood_packet_payload_repeated_request",
                "malware_indicators": "",
                "device_info": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            },
            "intrusion": {
                "source_port": 60123,
                "destination_port": 3389,
                "packet_length": 710,
                "protocol": "UDP",
                "packet_type": "Control",
                "traffic_type": "FTP",
                "severity": "Medium",
                "timestamp": "2024-01-09 11:33:01",
                "src_ip": "10.44.5.9",
                "dst_ip": "172.16.9.44",
                "payload": "auth_probe_sequence lateral movement attempt",
                "malware_indicators": "",
                "device_info": "Mozilla/5.0 (X11; Linux x86_64)",
            },
            "malware": {
                "source_port": 2231,
                "destination_port": 8080,
                "packet_length": 980,
                "protocol": "ICMP",
                "packet_type": "Data",
                "traffic_type": "DNS",
                "severity": "High",
                "timestamp": "2024-01-12 16:50:40",
                "src_ip": "192.168.32.5",
                "dst_ip": "91.203.11.66",
                "payload": "encoded beacon callback with suspicious binary chunk",
                "malware_indicators": "IoC",
                "device_info": "Mozilla/5.0 (Android 13; Mobile)",
            },
        }

        default_state = {
            "source_port": 50000,
            "destination_port": 443,
            "packet_length": 900,
            "protocol": "TCP",
            "packet_type": "Data",
            "traffic_type": "HTTP",
            "severity": "Medium",
            "timestamp": "2024-01-01 12:00:00",
            "src_ip": "192.168.1.10",
            "dst_ip": "8.8.8.8",
            "payload": "normal_payload",
            "malware_indicators": "",
            "device_info": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }
        for k, v in default_state.items():
            if k not in st.session_state:
                st.session_state[k] = v

        form_col, preset_col = st.columns([4, 1])
        with preset_col:
            st.markdown("#### Quick Profiles")
            if st.button("Load DDoS", use_container_width=True):
                for k, v in presets["ddos"].items():
                    st.session_state[k] = v
            if st.button("Load Intrusion", use_container_width=True):
                for k, v in presets["intrusion"].items():
                    st.session_state[k] = v
            if st.button("Load Malware", use_container_width=True):
                for k, v in presets["malware"].items():
                    st.session_state[k] = v
            st.caption("Use a profile, then click Predict.")

        with form_col:
            c1, c2, c3 = st.columns(3)

            with c1:
                source_port = st.number_input("Source Port", min_value=0, max_value=65535, key="source_port")
                destination_port = st.number_input("Destination Port", min_value=0, max_value=65535, key="destination_port")
                packet_length = st.number_input("Packet Length", min_value=0, key="packet_length")
                protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"], key="protocol")

            with c2:
                packet_type = st.selectbox("Packet Type", ["Data", "Control", "Unknown"], key="packet_type")
                traffic_type = st.selectbox("Traffic Type", ["HTTP", "DNS", "FTP", "SMTP", "Unknown"], key="traffic_type")
                severity = st.selectbox("Severity Level", ["Low", "Medium", "High"], key="severity")
                timestamp = st.text_input("Timestamp", key="timestamp")

            with c3:
                src_ip = st.text_input("Source IP Address", key="src_ip")
                dst_ip = st.text_input("Destination IP Address", key="dst_ip")
                payload = st.text_area("Payload Data", key="payload")
                malware_indicators = st.text_input("Malware Indicators", key="malware_indicators")

            device_info = st.text_input(
                "Device Information",
                key="device_info",
            )

        if st.button("Predict Attack Class", type="primary"):
            manual_df = pd.DataFrame(
                [
                    {
                        "Source Port": source_port,
                        "Destination Port": destination_port,
                        "Protocol": protocol,
                        "Packet Length": packet_length,
                        "Packet Type": packet_type,
                        "Traffic Type": traffic_type,
                        "Severity Level": severity,
                        "Timestamp": timestamp,
                        "Source IP Address": src_ip,
                        "Destination IP Address": dst_ip,
                        "Payload Data": payload,
                        "Malware Indicators": malware_indicators,
                        "Device Information": device_info,
                    }
                ]
            )

            pred_df = predict_with_model(manual_df, model, class_names, feature_columns)
            pred_idx = pred_df["Predicted Class"].map({name: i for i, name in enumerate(class_names)})
            y_gen_idx = generate_target_from_raw(manual_df)
            generated_labels = index_to_class(y_gen_idx, class_names)

            st.success("Prediction complete")

            card1, card2 = st.columns(2)
            with card1:
                st.markdown('<div class="soft-card"><div class="section-title">Predicted</div>'
                            f'<b>{pred_df.loc[0, "Predicted Class"]}</b><br>'
                            f'Confidence: <b>{pred_df.loc[0, "Confidence"]:.4f}</b></div>',
                            unsafe_allow_html=True)
            with card2:
                st.markdown('<div class="soft-card"><div class="section-title">Generated</div>'
                            f'<b>{generated_labels.iloc[0]}</b><br>'
                            f'Rule-based label from input features</div>',
                            unsafe_allow_html=True)

            agreement = float((pred_idx.iloc[0] == y_gen_idx.iloc[0]))
            st.metric("Predicted vs Generated Agreement", f"{agreement:.0%}")

            st.dataframe(pred_df[["Predicted Class", "Confidence"] + [f"P({c})" for c in class_names if f"P({c})" in pred_df.columns]], use_container_width=True)

            render_distribution_and_confidence(pred_df, generated_labels)

    with tab_csv:
        st.subheader("Batch Prediction")
        st.caption("Upload many rows, score them in one pass, then export the results.")
        st.caption(
            "Upload either model-ready features (all trained columns) or raw telemetry columns similar to your dataset."
        )

        template_df = pd.DataFrame(
            [
                {
                    "Timestamp": "2024-01-01 12:00:00",
                    "Source IP Address": "192.168.1.10",
                    "Destination IP Address": "8.8.8.8",
                    "Source Port": 50000,
                    "Destination Port": 443,
                    "Protocol": "TCP",
                    "Packet Length": 900,
                    "Packet Type": "Data",
                    "Traffic Type": "HTTP",
                    "Payload Data": "normal_payload",
                    "Malware Indicators": "",
                    "Severity Level": "Medium",
                    "Device Information": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                }
            ]
        )
        st.download_button(
            "Download CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(input_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                pred_df = predict_with_model(input_df, model, class_names, feature_columns)
                st.success(f"Predictions generated for {len(pred_df)} rows")
                st.dataframe(pred_df.head(50), use_container_width=True)

                pred_label_to_idx = {name: idx for idx, name in enumerate(class_names)}
                pred_idx = pred_df["Predicted Class"].map(pred_label_to_idx)

                y_gen_idx = generate_target_from_raw(input_df)
                generated_labels = index_to_class(y_gen_idx, class_names)

                agreement_valid = y_gen_idx.notna() & pred_idx.notna()
                agreement_rate = (
                    (pred_idx[agreement_valid].astype(int) == y_gen_idx[agreement_valid].astype(int)).mean()
                    if agreement_valid.sum() > 0
                    else np.nan
                )

                pred_top = pred_df["Predicted Class"].value_counts().index[0]
                gen_top = generated_labels.value_counts().index[0]

                card1, card2 = st.columns(2)
                with card1:
                    st.markdown('<div class="soft-card"><div class="section-title">Predicted</div>'
                                f'Dominant class: <b>{pred_top}</b><br>'
                                f'Rows: <b>{len(pred_df)}</b></div>',
                                unsafe_allow_html=True)
                with card2:
                    st.markdown('<div class="soft-card"><div class="section-title">Generated</div>'
                                f'Dominant class: <b>{gen_top}</b><br>'
                                f'Rows: <b>{len(generated_labels)}</b></div>',
                                unsafe_allow_html=True)

                if not np.isnan(agreement_rate):
                    st.metric("Predicted vs Generated Agreement", f"{agreement_rate:.2%}")

                st.markdown("### Target Comparison")
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Against Original Target**")
                    if "Attack Type" in input_df.columns:
                        y_orig = input_df["Attack Type"].apply(map_attack_type_to_3class)
                        valid = y_orig.notna() & pred_idx.notna()
                        if valid.sum() > 0:
                            y_true = y_orig[valid].astype(int)
                            y_hat = pred_idx[valid].astype(int)
                            acc_orig = accuracy_score(y_true, y_hat)
                            st.metric("Accuracy (Original)", f"{acc_orig:.4f}")
                            st.text(classification_report(y_true, y_hat, target_names=class_names, digits=3))
                        else:
                            st.info("`Attack Type` exists but no rows matched the 3-class mapping.")
                    else:
                        st.info("Upload a CSV with `Attack Type` to compare against the original target.")

                with c2:
                    st.markdown("**Against Generated Target**")
                    required_raw = {
                        "Timestamp", "Source Port", "Destination Port", "Protocol", "Packet Length",
                        "Packet Type", "Traffic Type", "Payload Data", "Malware Indicators", "Severity Level",
                    }
                    if required_raw.issubset(set(input_df.columns)):
                        y_gen = generate_target_from_raw(input_df)
                        valid = y_gen.notna() & pred_idx.notna()
                        if valid.sum() > 0:
                            y_true = y_gen[valid].astype(int)
                            y_hat = pred_idx[valid].astype(int)
                            acc_gen = accuracy_score(y_true, y_hat)
                            st.metric("Accuracy (Generated)", f"{acc_gen:.4f}")
                            st.text(classification_report(y_true, y_hat, target_names=class_names, digits=3))
                        else:
                            st.info("Generated target comparison had no valid rows.")
                    else:
                        st.info("Upload raw telemetry columns to compare against the generated target.")

                render_distribution_and_confidence(pred_df, generated_labels)

                st.download_button(
                    "Download Predictions CSV",
                    data=pred_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
