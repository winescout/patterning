## FEATURE:

A system to analyze user-provided E-mini futures trading videos, segment them by topic, and dynamically recommend relevant video slices based on real-time market price action using a Perceptually Important Point (PIP) algorithm. The initial implementation will be a Python-based CLI application, with a future goal of a GUI interface and potential re-implementation in Rust or another suitable language.

**Core Sub-features:**

*   **User-Driven Video Ingestion & Processing Pipeline:**
    *   Provide a mechanism for users to securely upload or specify the local paths to their MP4 video files.
    *   For each user-provided video:
        *   Extract audio tracks.
        *   Transcribe audio content into text.
        *   Apply Natural Language Processing (NLP) techniques to identify key topics and segment videos into logical, topic-based slices.
*   **Video Cataloging & Metadata Management (User-Specific):**
    *   Store comprehensive metadata for each video slice, including video ID (unique per user), original file path (or reference), start/end timestamps, identified topics, and relevant keywords. This will likely involve a database, potentially with user-specific indexing.
*   **Live Market Data Integration:**
    *   Establish a connection to a real-time E-mini futures market data feed.
*   **Perceptually Important Point (PIP) Algorithm Implementation:**
    *   Develop or integrate an algorithm to detect significant and "perceptually important" points within the live market data stream.
*   **Recommendation Engine:**
    *   Develop logic to intelligently match the identified PIPs from live market data with the most relevant video slices from the user's catalog.
    *   Present a ranked list of recommended video slices to the user.

## EXAMPLES:

*   **CLI Interaction:** A command-line interface where a user can:
    *   Initiate the video ingestion process (e.g., `python main.py ingest --video-path /path/to/my/trading_video.mp4`).
    *   Start the live market data analysis and receive recommendations (e.g., `python main.py analyze-live-market`).
    *   Output: A list of relevant video slices (e.g., `video_id: timestamp_start - timestamp_end, topic: "Market Open Strategy"`).
*   **Future UI (Conceptual):** Imagine a local application where users can drag-and-drop their video files for processing, and then a dashboard displays live market charts alongside dynamically updating video recommendations.

## DOCUMENTATION:

To successfully implement this, we'll need to research and potentially integrate with the following:

*   **Video/Audio Processing:** `ffmpeg` (for command-line video manipulation), Python libraries like `moviepy` or `pydub` for programmatic audio extraction.
*   **Speech-to-Text (STT):** Cloud-based APIs (e.g., Google Cloud Speech-to-Text, OpenAI Whisper API, AssemblyAI) or local STT models for transcription accuracy and efficiency. Users would likely need to configure their own API keys if cloud services are used.
*   **Natural Language Processing (NLP):** Python libraries such as `spaCy`, `NLTK`, `Gensim`, or `scikit-learn` for text cleaning, topic modeling (e.g., LDA, BERTopic), and keyword extraction.
*   **Database Systems:** Consider PostgreSQL or SQLite for structured storage of video metadata.
*   **Market Data APIs:** Identify and integrate with a reliable E-mini futures market data provider (e.g., Interactive Brokers, Alpaca, Polygon.io). Users would need to provide their own API credentials.
*   **PIP Algorithms:** Research existing academic literature and practical implementations of Perceptually Important Point algorithms in financial time series analysis.

## OTHER CONSIDERATIONS:

*   **User Data Privacy & Security:** Since users will be providing their own video files, the system must be designed with privacy in mind. No video content should be transmitted or stored externally without explicit user consent. Processing should ideally happen locally on the user's machine.
*   **User Experience for Ingestion:** The ingestion process needs to be clear, robust, and provide good feedback to the user (e.g., progress indicators, error messages for invalid files).
*   **Scalability:** The system should be designed to handle a potentially large volume of video data per user and continuous live market data.
*   **Accuracy & Latency:** The accuracy of transcription, topic modeling, and PIP detection will be critical for the system's utility. For live recommendations, low latency in data processing and matching is paramount.
*   **Computational Resources:** Processing videos and live data can be resource-intensive; users will need adequate hardware.
*   **Error Handling:** Robust error handling will be necessary for external API integrations (STT, market data) and local file operations.
*   **Dependency Management:** Clearly document all dependencies and provide easy installation instructions for users.
