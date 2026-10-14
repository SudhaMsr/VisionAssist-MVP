"""VisionAssist web UI. Run from the repo root:  streamlit run app.py

Analyses an MP4 with the existing pipeline (YOLOv8n detector, alert rules,
single speech channel with urgent interruption) and shows the detections,
the prioritised spoken alerts, the audio, and downloadable outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from vision_assist.config import REPO_ROOT
from vision_assist.webapp import (
    alert_rows,
    analyse_video,
    detection_rows,
    frame_caption,
    list_videos,
    new_run_dir,
    save_upload,
    video_dirs,
    zip_outputs,
)

st.set_page_config(page_title="VisionAssist", layout="wide")
st.title("VisionAssist: obstacle alerts from video")
st.warning(
    "Experimental prototype. It can miss or misclassify obstacles, and silence does not mean the way is "
    "clear. Never use it as the sole means of navigation. \"left / centre / right\" is the third of the "
    "image, not a direction, and box size is not a distance estimate."
)

with st.sidebar:
    st.header("Settings")
    fps = st.select_slider("Frames analysed per second", options=[0.5, 1.0, 2.0, 3.0, 5.0], value=2.0)
    limit = st.number_input("Maximum frames to analyse", min_value=1, max_value=1000, value=120, step=10,
                            help="Keeps run time bounded: about 0.1 s per frame on a CPU.")
    conf = st.slider("Detection confidence threshold", 0.10, 0.90, 0.35, 0.05)
    max_alerts = st.slider("Objects named per message", 1, 3, 2)
    cooldown = st.slider("Seconds before repeating the same alert", 0.0, 20.0, 6.0, 0.5)
    urgent = st.checkbox("Urgent alerts interrupt lower-priority speech", value=True)

st.subheader("1. Choose a video")
source = st.radio("Video source", ["Upload an MP4", "Choose an existing MP4"], horizontal=True)
upload, chosen = None, None
if source == "Upload an MP4":
    upload = st.file_uploader("MP4 file", type=["mp4"])
    if upload is not None:
        st.video(upload)
else:
    videos = list_videos()
    if videos:
        chosen = st.selectbox("MP4 file", videos, format_func=lambda p: p.name)
        st.video(str(chosen))
    else:
        folders = ", ".join(str(d) for d in video_dirs())
        st.info(f"No MP4 files found in {folders}. Create a sample with "
                "`python scripts/make_sample_video.py`, or upload one.")

st.subheader("2. Analyse")
ready = upload is not None or chosen is not None
if st.button("Analyse video", type="primary", disabled=not ready):
    name = upload.name if upload is not None else chosen.name
    run_dir = new_run_dir(name)
    bar = st.progress(0.0, text="Starting...")
    try:
        video = save_upload(upload.name, upload.getvalue(), run_dir) if upload is not None else chosen
        summary = analyse_video(
            video, run_dir, fps=fps, limit=int(limit), conf=conf, max_alerts=max_alerts,
            cooldown_s=cooldown, urgent_interrupts=urgent,
            progress=lambda i, n: bar.progress(i / n, text=f"Analysing frame {i} of {n}"),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        bar.empty()
        st.error(f"Analysis failed: {exc}")
        st.session_state.pop("run_dir", None)
    else:
        bar.progress(1.0, text="Done")
        st.session_state["run_dir"] = str(run_dir)
        st.session_state["video_name"] = name

run_dir = Path(st.session_state["run_dir"]) if "run_dir" in st.session_state else None
if run_dir and (run_dir / "stream_results.json").is_file():
    summary = json.loads((run_dir / "stream_results.json").read_text())
    m, u = summary["metrics"], summary["metrics"]["urgent"]
    st.subheader(f"3. Results for {st.session_state.get('video_name', 'video')}")
    try:
        st.caption(f"Saved in {run_dir.relative_to(REPO_ROOT)}")
    except ValueError:
        st.caption(f"Saved in {run_dir}")

    cols = st.columns(6)
    cols[0].metric("Frames analysed", summary["frames"])
    cols[1].metric("Messages spoken", m["messages_spoken"])
    cols[2].metric("Messages per minute", m["messages_per_min"])
    cols[3].metric("Time spent speaking", f"{m['speaking_pct']}%")
    cols[4].metric("Urgent hazards announced", f"{u['announced']} of {u['high_tier_episodes']}")
    cols[5].metric("Overlapping speech", m["overlapping_messages"])

    tab_av, tab_alerts, tab_det, tab_frames, tab_dl = st.tabs(
        ["Video and audio", "Spoken alerts", "Detections", "Frame by frame", "Downloads"])
    with tab_av:
        st.video(str(run_dir / "vision_assist_stream.mp4"))
        st.caption("Annotated frames with the spoken alerts. Audio only:")
        st.audio(str(run_dir / "vision_assist_stream.wav"))
    with tab_alerts:
        rows = alert_rows(summary)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Nothing was spoken. No alert-worthy object was detected above the threshold.")
        st.caption(f"Messages cut off by an urgent alert: {m['urgent_interruptions']}. "
                   f"Alerts that left the view while other speech was playing: {m['dropped_while_busy']}.")
    with tab_det:
        det = pd.DataFrame(detection_rows(summary))
        if det.empty:
            st.info("No objects were detected.")
        else:
            classes = st.multiselect("Show classes", sorted(det["class"].unique()))
            st.dataframe(det[det["class"].isin(classes)] if classes else det, hide_index=True,
                         width="stretch")
    with tab_frames:
        frames = summary["per_frame"]
        i = st.slider("Frame", 1, len(frames), 1) if len(frames) > 1 else 1
        frame = frames[i - 1]
        st.image(str(run_dir / "frames" / f"frame_{i:04d}.jpg"),
                 caption=f"t = {frame['t_s']} s. {frame_caption(frame)}")
    with tab_dl:
        files = [("Video (MP4)", "vision_assist_stream.mp4", "video/mp4"),
                 ("Audio (WAV)", "vision_assist_stream.wav", "audio/wav"),
                 ("Transcript (TXT)", "transcript.txt", "text/plain"),
                 ("Full results (JSON)", "stream_results.json", "application/json")]
        for label, fname, mime in files:
            st.download_button(label, (run_dir / fname).read_bytes(), file_name=fname, mime=mime, key=fname)
        st.download_button("Everything, including annotated frames (ZIP)", zip_outputs(run_dir),
                           file_name=f"{run_dir.name}.zip", mime="application/zip", key="zip")
