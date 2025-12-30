"""FluidSynth-based audio rendering for chords."""

from typing import Union, List

import numpy as np
import fluidsynth

from .voicing import NVoiceVoicing, SATBVoicing

# Import VoiceEvent for type hints (avoid circular import)
try:
    from src.data_generation.voicing.non_chord_tones import VoiceEvent
except ImportError:
    VoiceEvent = None  # Will use duck typing if import fails


class FluidSynthRenderer:
    """Render voicings to audio using FluidSynth."""

    def __init__(
        self,
        soundfont_path: str,
        sample_rate: int = 44100,
        choir_program: int = 52,  # 52 = Choir Aahs in General MIDI
    ):
        """
        Initialize the FluidSynth renderer.

        Args:
            soundfont_path: Path to .sf2 soundfont file
            sample_rate: Audio sample rate in Hz
            choir_program: MIDI program number for choir sound (52=Choir Aahs, 53=Voice Oohs)
        """
        self.sample_rate = sample_rate
        self.choir_program = choir_program

        # Initialize FluidSynth
        self.fs = fluidsynth.Synth(samplerate=float(sample_rate))
        self.sfid = self.fs.sfload(soundfont_path)

        if self.sfid == -1:
            raise RuntimeError(f"Failed to load soundfont: {soundfont_path}")

        # Set up 16 channels (supports up to 16 voices) with choir sound
        for channel in range(16):
            self.fs.program_select(channel, self.sfid, 0, choir_program)

    def render_chord(
        self,
        voicing: Union[SATBVoicing, NVoiceVoicing],
        duration_sec: float = 3.0,
        velocity: int = 80,
        release_sec: float = 0.5,
    ) -> np.ndarray:
        """
        Render a chord voicing to audio.

        Args:
            voicing: SATB or N-voice voicing to render
            duration_sec: Duration of sustained chord in seconds
            velocity: MIDI velocity (0-127)
            release_sec: Additional time after note-off for release tail

        Returns:
            Numpy array of audio samples (mono, float32, -1 to 1)
        """
        notes = voicing.to_midi_notes()

        # Start all notes (one per channel to avoid voice stealing, wrap at 16)
        for i, note in enumerate(notes):
            channel = i % 16
            self.fs.noteon(channel, note, velocity)

        # Render sustain portion
        sustain_samples = int(duration_sec * self.sample_rate)
        audio = self._get_samples(sustain_samples)

        # Stop all notes
        for i, note in enumerate(notes):
            channel = i % 16
            self.fs.noteoff(channel, note)

        # Render release tail
        release_samples = int(release_sec * self.sample_rate)
        release_audio = self._get_samples(release_samples)

        # Combine sustain and release
        full_audio = np.concatenate([audio, release_audio])

        return full_audio

    def _get_samples(self, num_samples: int) -> np.ndarray:
        """
        Get audio samples from FluidSynth.

        Args:
            num_samples: Number of samples to render

        Returns:
            Mono audio as numpy array (float32, normalized to -1 to 1)
        """
        # FluidSynth returns stereo interleaved samples
        samples = self.fs.get_samples(num_samples)

        # Convert to numpy array
        audio = np.array(samples, dtype=np.float32)

        # Convert from stereo interleaved to mono
        # Samples are [L0, R0, L1, R1, ...], reshape and average
        if len(audio) > 0:
            audio = audio.reshape(-1, 2)
            audio = audio.mean(axis=1)

        # Normalize to -1 to 1 (FluidSynth outputs 16-bit range)
        audio = audio / 32768.0

        return audio

    def render_chord_with_events(
        self,
        voice_events: List[List],  # List[List[VoiceEvent]]
        duration_sec: float,
        velocity: int = 80,
        release_sec: float = 0.5,
    ) -> np.ndarray:
        """
        Render time-varying voice events to audio.

        This method supports non-chord tones by rendering voice events
        that change pitch over time. Each voice has a list of events
        with start/end times and pitches.

        Args:
            voice_events: List of event lists, one per voice. Each event
                         has pitch (int), start_time (float), end_time (float).
            duration_sec: Total duration of the chord in seconds
            velocity: MIDI velocity (0-127)
            release_sec: Additional time after note-off for release tail

        Returns:
            Numpy array of audio samples (mono, float32, -1 to 1)
        """
        total_samples = int(duration_sec * self.sample_rate)
        audio = np.zeros(total_samples, dtype=np.float32)

        # Collect all event boundaries to determine render segments
        boundaries = {0.0, duration_sec}
        for voice_idx, events in enumerate(voice_events):
            for event in events:
                boundaries.add(event.start_time)
                boundaries.add(min(event.end_time, duration_sec))

        # Sort boundaries into segments
        segments = sorted(boundaries)

        # Track active notes per channel
        active_notes = {}  # channel -> current_note

        # Render each segment
        for seg_idx in range(len(segments) - 1):
            seg_start = segments[seg_idx]
            seg_end = segments[seg_idx + 1]
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                continue

            # Determine which notes should be active in this segment
            desired_notes = {}  # channel -> note
            for voice_idx, events in enumerate(voice_events):
                channel = voice_idx % 16
                for event in events:
                    # Check if event is active during this segment
                    if event.start_time <= seg_start < event.end_time:
                        desired_notes[channel] = event.pitch
                        break

            # Stop notes that should no longer be active
            for channel, note in list(active_notes.items()):
                if channel not in desired_notes or desired_notes[channel] != note:
                    self.fs.noteoff(channel, note)
                    del active_notes[channel]

            # Start notes that should be active
            for channel, note in desired_notes.items():
                if channel not in active_notes or active_notes[channel] != note:
                    self.fs.noteon(channel, note, velocity)
                    active_notes[channel] = note

            # Render this segment
            seg_samples = int(seg_duration * self.sample_rate)
            if seg_samples > 0:
                seg_audio = self._get_samples(seg_samples)
                start_sample = int(seg_start * self.sample_rate)
                end_sample = min(start_sample + len(seg_audio), total_samples)
                audio[start_sample:end_sample] = seg_audio[:end_sample - start_sample]

        # Stop all remaining notes
        for channel, note in active_notes.items():
            self.fs.noteoff(channel, note)

        # Render release tail
        release_samples = int(release_sec * self.sample_rate)
        release_audio = self._get_samples(release_samples)

        # Combine sustain and release
        full_audio = np.concatenate([audio, release_audio])

        return full_audio

    def cleanup(self):
        """Clean up FluidSynth resources."""
        self.fs.delete()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
