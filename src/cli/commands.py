import os
import tempfile
import uuid  # For unique IDs
import datetime  # For timestamps
from video_ingestion.audio_extractor import extract_audio
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
        output_markdown_path = os.path.join(
            os.getcwd(), f"spike_output_{uuid.uuid4().hex[:8]}.md"
        )

        try:
            extract_audio(video_path, audio_output_path)
            full_transcript = transcribe_audio(audio_output_path)

            topics_data = extract_topics_with_timestamps(
                full_transcript, video_duration
            )

            with open(output_markdown_path, "w") as f:
                f.write(
                    f"# Video Ingestion Spike Report for: {os.path.basename(video_path)}\n\n"
                )
                f.write(f"**Date:** {datetime.datetime.now().isoformat()}\n\n")
                f.write("## Full Transcript\n\n")
                f.write(full_transcript)
                f.write("\n\n## Identified Topics with Timestamps\n\n")
                for i, topic_info in enumerate(topics_data):
                    f.write(f"{i + 1}. **Topic:** {topic_info['topic']}\n")
                    f.write(
                        f"   **Time:** {topic_info['start_time']:.2f}s - {topic_info['end_time']:.2f}s\n"
                    )
                    f.write(f'   **Segment:** "{topic_info["text_segment"]}"\n\n')

            print(f"Spike successful! Report generated at: {output_markdown_path}")

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
