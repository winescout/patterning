import os
import tempfile
import uuid  # For unique IDs
import datetime  # For timestamps
from video_ingestion.audio_extractor import extract_audio
from video_ingestion.image_extractor import extract_screenshots
from video_ingestion.transcriber import transcribe_audio
from video_ingestion.topic_extractor import extract_topics_with_timestamps
from moviepy.editor import VideoFileClip  # For getting video duration


def spike_ingest_command(args):
    """
    CLI command for the video ingestion spike.
    """
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Get video duration
    try:
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        clip.close()
        print(f"Video duration: {video_duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting video duration: {e}. Cannot proceed with timestamping.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_output_path = os.path.join(tmpdir, "extracted_audio.wav")
        output_dir = os.path.join(os.getcwd(), "output")
        screenshots_dir = os.path.join(output_dir, "screenshots")
        reports_dir = os.path.join(output_dir, "reports")
        output_markdown_path = os.path.join(reports_dir, f"spike_output_{uuid.uuid4().hex[:8]}.md")
        
        # Create output directories
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        try:
            extract_audio(video_path, audio_output_path)
            screenshot_files = extract_screenshots(video_path, screenshots_dir, interval_seconds=4)
            full_transcript = transcribe_audio(audio_output_path)

            # Note: extract_topics_with_timestamps now returns a dict of {keyword: [timestamps]}
            # We need to create a fake Whisper result for compatibility
            fake_whisper_result = {
                "text": full_transcript,
                "segments": [
                    {
                        "text": full_transcript,
                        "words": []  # Empty for now, real implementation would have word-level timestamps
                    }
                ]
            }
            
            topics_data = extract_topics_with_timestamps(fake_whisper_result)

            with open(output_markdown_path, "w") as f:
                f.write(
                    f"# Video Ingestion Spike Report for: {os.path.basename(video_path)}\n\n"
                )
                f.write(f"**Date:** {datetime.datetime.now().isoformat()}\n\n")
                
                f.write("## Extracted Screenshots\n\n")
                f.write(f"Screenshots captured every 4 seconds ({len(screenshot_files)} total):\n\n")
                for screenshot in screenshot_files:
                    rel_path = os.path.relpath(screenshot, reports_dir)
                    f.write(f"- `{rel_path}`\n")
                f.write("\n")
                
                f.write("## Full Transcript\n\n")
                f.write(full_transcript)
                f.write("\n\n## Identified Keywords with Timestamps\n\n")
                if topics_data:
                    for keyword, timestamps in topics_data.items():
                        f.write(f"**{keyword.title()}**: ")
                        timestamp_strs = [f"{ts['start']:.1f}s-{ts['end']:.1f}s" for ts in timestamps]
                        f.write(", ".join(timestamp_strs))
                        f.write("\n\n")
                else:
                    f.write("No keywords identified in the transcript.\n\n")

            print(f"Spike successful!")
            print(f"Report generated at: {output_markdown_path}")
            print(f"Screenshots saved to: {screenshots_dir}")
            print(f"All outputs saved to: {output_dir}")

        except RuntimeError as e:
            print(f"Spike failed: {e}")
        finally:
            # Clean up temporary audio file if it exists
            if os.path.exists(audio_output_path):
                os.remove(audio_output_path)


def add_spike_commands(parser):
    """Adds spike-specific commands to the CLI parser."""
    spike_parser = parser.add_parser(
        "spike-ingest",
        help="Run an end-to-end spike for video ingestion technology validation.",
    )
    spike_parser.add_argument(
        "--video-path", required=True, help="Path to the MP4 video file for the spike."
    )
    spike_parser.set_defaults(func=spike_ingest_command)
