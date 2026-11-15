import tempfile
from pathlib import Path

import streamlit as st

import clipper

# ================== APNI BRANDING YAHAN BADLO ==================
APP_NAME = "Dark Clip Maker"          # <-- apne app ka naam
OWNER_NAME = "DarkDeathEvil"             # <-- apna naam
TAGLINE = "YouTube link dalo, best clips lo, Reels/Shorts par upload karo"
LOGO_FILE = Path(__file__).parent / "logo.png"   # <-- apna logo isi naam se (logo.png) folder mein rakho
# ================================================================

logo_exists = LOGO_FILE.exists()
st.set_page_config(page_title=APP_NAME, page_icon=str(LOGO_FILE) if logo_exists else "🎬", layout="wide")

head1, head2 = st.columns([1, 8])
with head1:
    if logo_exists:
        st.image(str(LOGO_FILE), width=90)
with head2:
    st.title(APP_NAME)
    st.caption(f"{TAGLINE}  |  by {OWNER_NAME}")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                            help="Khali chhodo to bina AI ke simple scoring use hogi.")
    model = st.text_input("AI model", value="claude-sonnet-5-5")
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2)
    device = st.selectbox("Device", ["cpu", "cuda"], help="NVIDIA GPU ho to cuda")
    n_clips = st.slider("Kitne clips", 1, 10, 5)
    min_d, max_d = st.slider("Clip length (seconds)", 15, 90, (30, 60))
    captions = st.checkbox("Captions lagao", True)
    face_track = st.checkbox("Chehra track karo", True)
    cookies_choice = st.selectbox("Browser login (age-restricted videos ke liye)",
                                  ["Nahi", "firefox", "edge", "chrome"],
                                  help="Sirf tab chuno jab video age-restricted ho. Us browser mein YouTube par login hona chahiye.")

url = st.text_input("YouTube link")

if st.button("Generate Clips", type="primary") and url:
    work = Path(tempfile.mkdtemp(prefix="clips_"))
    try:
        with st.status("Kaam ho raha hai...", expanded=True) as status:
            st.write("1/4 Video download ho rahi hai...")
            video, title = clipper.download_video(url, work, None if cookies_choice == "Nahi" else cookies_choice)
            st.write("2/4 Awaaz text mein badal rahi hai (thora time lagega)...")
            segments = clipper.transcribe(video, whisper_size, device)
            if not segments:
                st.error("Video mein koi awaaz/speech nahi mili.")
                st.stop()
            st.write("3/4 Best moments dhundh rahe hain...")
            if api_key:
                clips = clipper.pick_clips_ai(segments, api_key, model, n_clips, min_d, max_d)
            else:
                clips = clipper.pick_clips_heuristic(segments, n_clips, min_d, max_d)
            clips.sort(key=lambda c: c["score"], reverse=True)
            st.write("4/4 Clips ban rahe hain...")
            results = []
            for i, c in enumerate(clips, 1):
                path = clipper.render_clip(video, segments, c, work, i, captions, face_track)
                results.append((c, path))
            status.update(label="Ho gaya!", state="complete")
        st.session_state["results"] = results
    except Exception as e:
        st.error(f"Error: {clipper._clean(e)}")

for i, (c, path) in enumerate(st.session_state.get("results", []), 1):
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(str(path))
    with col2:
        st.subheader(f"#{i}  {c['title']}")
        st.write(f"**Score:** {c['score']}  |  **Time:** {c['start']}s → {c['end']}s")
        st.write(c["reason"])
        st.download_button("Download MP4", Path(path).read_bytes(),
                           file_name=f"clip_{i}.mp4", mime="video/mp4", key=f"dl{i}")

st.divider()
st.caption(f"© {OWNER_NAME} | {APP_NAME}")
