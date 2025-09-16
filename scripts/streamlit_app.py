import time
import io
import tempfile
from pathlib import Path
import threading
import queue

import cv2
import numpy as np
import requests
import streamlit as st


st.set_page_config(page_title="Safe Watch - Vision Client", layout="wide")


# ---------------------------
# Config
# ---------------------------
DEFAULT_URL = "https://8000-dep-01k58ypwee5csr6fgb7kdgs0vz-d.cloudspaces.litng.ai/vision/infer"
DEFAULT_TOKEN = ""
POLL_INTERVAL_SECONDS = 10


def post_frame(url: str, token: str, frame_bgr: np.ndarray, prompt_variant: str) -> requests.Response:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    files = {"image": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
    data = {"prompt_variant": prompt_variant}
    return requests.post(url, headers=headers, files=files, data=data, timeout=60)


def color_for_violence(violence: bool) -> str:
    return "#f44336" if violence else "#4caf50"  # red / green


def poster_thread(
    url: str,
    token: str,
    prompt_variant: str,
    send_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Send frames in the background so the UI stays responsive."""
    while not stop_event.is_set():
        try:
            frame, err = send_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if err is not None:
            try:
                if result_queue.full():
                    result_queue.get_nowait()
                result_queue.put_nowait((None, err))
            except Exception:
                pass
            continue
        try:
            resp = post_frame(url, token, frame, prompt_variant)
            payload = (resp, None)
        except Exception as ex:
            payload = (None, f"Request failed: {ex}")
        try:
            if result_queue.full():
                result_queue.get_nowait()
            result_queue.put_nowait(payload)
        except Exception:
            pass


def reader_thread_from_video(
    video_path: Path,
    frame_queue: queue.Queue,
    preview_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        frame_queue.put((None, "Unable to open video"))
        return
    last_sent = 0.0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # End of video
                time.sleep(0.1)
                continue
            # Update latest frame for live preview via queue (drop old)
            try:
                if preview_queue.full():
                    preview_queue.get_nowait()
                preview_queue.put_nowait(frame)
            except Exception:
                pass
            now = time.time()
            if now - last_sent >= POLL_INTERVAL_SECONDS:
                frame_queue.put((frame, None))
                last_sent = now
            # Reduce CPU usage
            time.sleep(0.01)
    finally:
        cap.release()


def reader_thread_from_webcam(
    device_index: int,
    frame_queue: queue.Queue,
    preview_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        frame_queue.put((None, f"Unable to open webcam index {device_index}"))
        return
    last_sent = 0.0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # Update latest frame for live preview via queue (drop old)
            try:
                if preview_queue.full():
                    preview_queue.get_nowait()
                preview_queue.put_nowait(frame)
            except Exception:
                pass
            now = time.time()
            if now - last_sent >= POLL_INTERVAL_SECONDS:
                frame_queue.put((frame, None))
                last_sent = now
            time.sleep(0.01)
    finally:
        cap.release()


def main() -> None:
    st.title("Safe Watch - Vision Client")

    with st.sidebar:
        st.header("Connection")
        url = st.text_input("Vision Endpoint URL", value=DEFAULT_URL)
        token = st.text_input("Bearer Token (optional)", value=DEFAULT_TOKEN, type="password")
        prompt_variant = st.selectbox("Prompt Variant", ["fewshot", "default"], index=0)
        st.header("Source")
        mode = st.radio("Choose input mode", ["Upload video", "Webcam"], index=0)
        device_index = st.number_input("Webcam index", min_value=0, value=0, step=1, help="0 is default camera")
        toast_enabled = st.checkbox("Toast notifications", value=True)

        st.divider()
        st.caption(f"Sending one frame every {POLL_INTERVAL_SECONDS} seconds")

    col_left, col_right = st.columns([1, 1])

    # State
    if "frame_queue" not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=10)
    if "preview_queue" not in st.session_state:
        st.session_state.preview_queue = queue.Queue(maxsize=1)
    if "send_queue" not in st.session_state:
        st.session_state.send_queue = queue.Queue(maxsize=2)
    if "result_queue" not in st.session_state:
        st.session_state.result_queue = queue.Queue(maxsize=2)
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if "reader_thread" not in st.session_state:
        st.session_state.reader_thread = None

    # Controls
    start_clicked = False
    stop_clicked = False

    if mode == "Upload video":
        uploaded = col_left.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"]) 
        start_clicked = col_left.button("Start", type="primary", disabled=uploaded is None)
        stop_clicked = col_left.button("Stop", disabled=False)
    else:
        col_left.markdown("Using webcam. Adjust index in the sidebar if needed.")
        start_clicked = col_left.button("Start", type="primary")
        stop_clicked = col_left.button("Stop")

    # Start/Stop reader threads
    if start_clicked and (st.session_state.reader_thread is None or not st.session_state.reader_thread.is_alive()):
        st.session_state.stop_event.clear()
        if mode == "Upload video":
            temp_dir = Path(tempfile.gettempdir())
            tmp_file = temp_dir / (uploaded.name if uploaded else f"uploaded_{int(time.time())}.mp4")
            if uploaded:
                with open(tmp_file, "wb") as f:
                    f.write(uploaded.getbuffer())
            reader = threading.Thread(
                target=reader_thread_from_video,
                args=(tmp_file, st.session_state.frame_queue, st.session_state.preview_queue, st.session_state.stop_event),
                daemon=True,
            )
        else:
            reader = threading.Thread(
                target=reader_thread_from_webcam,
                args=(int(device_index), st.session_state.frame_queue, st.session_state.preview_queue, st.session_state.stop_event),
                daemon=True,
            )
        reader.start()
        st.session_state.reader_thread = reader
        # Start poster thread once per session
        if "poster_thread" not in st.session_state or st.session_state.poster_thread is None or not st.session_state.poster_thread.is_alive():
            poster = threading.Thread(
                target=poster_thread,
                args=(url, token, prompt_variant, st.session_state.send_queue, st.session_state.result_queue, st.session_state.stop_event),
                daemon=True,
            )
            poster.start()
            st.session_state.poster_thread = poster

    if stop_clicked and st.session_state.reader_thread is not None:
        st.session_state.stop_event.set()
        st.session_state.reader_thread.join(timeout=1)
        st.session_state.reader_thread = None

    # Live panel
    frame_placeholder = col_left.empty()
    status_placeholder = col_right.empty()
    raw_placeholder = col_right.expander("Raw response", expanded=False)

    # Show preview when not running
    if st.session_state.reader_thread is None or not st.session_state.reader_thread.is_alive():
        if mode == "Webcam":
            cap_preview = cv2.VideoCapture(int(device_index))
            if cap_preview.isOpened():
                ret_prev, frame_prev = cap_preview.read()
                if ret_prev:
                    try:
                        prev_rgb = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(prev_rgb, channels="RGB", use_column_width=True)
                    except Exception:
                        pass
            cap_preview.release()
        elif mode == "Upload video":
            if 'uploaded' in locals() and uploaded is not None:
                col_left.video(uploaded)

    # Poll results and send
    try:
        while st.session_state.reader_thread is not None and st.session_state.reader_thread.is_alive():
            # Live preview from latest frame if available
            try:
                latest = st.session_state.preview_queue.get_nowait()
            except Exception:
                latest = None
            if latest is not None:
                try:
                    frame_rgb_live = cv2.cvtColor(latest, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb_live, channels="RGB", use_column_width=True)
                except Exception:
                    pass

            # Try to get a frame scheduled for sending (every 10s)
            try:
                frame, err = st.session_state.frame_queue.get(timeout=0.05)
            except queue.Empty:
                # Also check for response updates even if no frame to send
                try:
                    resp, req_err = st.session_state.result_queue.get_nowait()
                except Exception:
                    resp, req_err = None, None
                if resp is not None:
                    try:
                        if resp.status_code == 200:
                            data = resp.json()
                            is_violent = bool(data.get("violence_detected", False))
                            severity = data.get("severity_score")
                            scene = data.get("scene_description")
                            color = color_for_violence(is_violent)
                            bg_style = f"background-color:{color}20;padding:16px;border-radius:8px;border:1px solid {color}"
                            status_placeholder.markdown(
                                f"<div style='{bg_style}'>"
                                f"<h3 style='margin:0;color:{color}'>"
                                f"{'VIOLENCE DETECTED' if is_violent else 'NO VIOLENCE'}</h3>"
                                f"<p style='margin:4px 0'>Severity: {severity if severity is not None else 'N/A'}</p>"
                                f"<p style='margin:4px 0'>{scene or ''}</p>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            if toast_enabled:
                                try:
                                    st.toast("VIOLENCE DETECTED" if is_violent else "No violence", icon="⚠️" if is_violent else "✅")
                                except Exception:
                                    pass
                            with raw_placeholder:
                                st.json(data)
                        else:
                            status_placeholder.error(f"HTTP {resp.status_code}: {resp.text[:2000]}")
                    except Exception:
                        pass
                elif req_err is not None:
                    status_placeholder.error(req_err)
                time.sleep(0.02)
                continue

            if err is not None:
                status_placeholder.error(err)
                time.sleep(0.02)
                continue

            # Schedule sending in background
            try:
                if st.session_state.send_queue.full():
                    st.session_state.send_queue.get_nowait()
                st.session_state.send_queue.put_nowait((frame, None))
            except Exception:
                pass

            # Also consume any ready responses
            try:
                resp, req_err = st.session_state.result_queue.get_nowait()
            except Exception:
                resp, req_err = None, None
            if resp is not None:
                if resp.status_code == 200:
                    data = resp.json()
                    is_violent = bool(data.get("violence_detected", False))
                    severity = data.get("severity_score")
                    scene = data.get("scene_description")
                    color = color_for_violence(is_violent)
                    bg_style = f"background-color:{color}20;padding:16px;border-radius:8px;border:1px solid {color}"
                    status_placeholder.markdown(
                        f"<div style='{bg_style}'>"
                        f"<h3 style='margin:0;color:{color}'>"
                        f"{'VIOLENCE DETECTED' if is_violent else 'NO VIOLENCE'}</h3>"
                        f"<p style='margin:4px 0'>Severity: {severity if severity is not None else 'N/A'}</p>"
                        f"<p style='margin:4px 0'>{scene or ''}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if toast_enabled:
                        try:
                            st.toast("VIOLENCE DETECTED" if is_violent else "No violence", icon="⚠️" if is_violent else "✅")
                        except Exception:
                            pass
                    with raw_placeholder:
                        st.json(data)
                else:
                    status_placeholder.error(f"HTTP {resp.status_code}: {resp.text[:2000]}")
            elif req_err is not None:
                status_placeholder.error(req_err)

            time.sleep(0.02)
    except st.runtime.scriptrunner.StopException:  # type: ignore[attr-defined]
        pass


if __name__ == "__main__":
    main()


