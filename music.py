from pydub.generators import Sawtooth
from pydub.playback import play
import time
from pydub.generators import Sawtooth
from pydub.playback import play
from pydub import AudioSegment


def play_pure_note(frequency, duration_ms):
    # Generate a sine wave representing the pure note
    sine_wave = Sawtooth(frequency)

    # Convert the duration from milliseconds to seconds
    duration_seconds = duration_ms / 1000.0

    # Generate the audio segment for the pure note
    audio_segment = sine_wave.to_audio_segment(
        duration=duration_seconds * note_duration
    )

    # Play the audio
    play(audio_segment)


# Example usage: play a 440 Hz pure note for 1 second

piano_frequencies = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "Bb4": 466.16,
    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.25,
    "F5": 698.46,
    "G5": 783.99,
    "A5": 880.00,
}




def play_song(notes, note_duration, piano_frequencies):
    for note in notes:
        if note == "":
            time.sleep(note_duration / 1000)
        else:
            play_pure_note(piano_frequencies[note], note_duration)
    pass


note_duration = 500
super_mario_theme = [
    "E5",
    "E5",
    "E5",
    "C5",
    "E5",
    "G5",
    "G4",
    "C5",
    "G4",
    "E4",
    "A4",
    "B4",
    "Bb4",
    "A4",
    "G4",
    "E5",
    "G5",
    "A5",
    "F5",
    "G5",
    "E5",
    "C5",
    "D5",
    "B4",
    "C5",
    "G4",
    "E4",
    "A4",
    "B4",
    "Bb4",
    "A4",
    "G4",
    "E5",
    "G5",
    "A5",
    "F5",
    "G5",
    "E5",
    "C5",
    "D5",
    "B4",
    "C5",
    "G4",
    "E4",
    "E5",
    "E5",
    "E5",
    "C5",
    "E5",
    "G5",
    "G4",
    "C5",
    "G4",
    "E4",
    "A4",
    "B4",
    "Bb4",
    "A4",
    "G4",
    "E5",
    "G5",
    "A5",
    "F5",
    "G5",
    "E5",
    "C5",
    "D5",
    "B4",
    "C5",
    "G4",
    "E4",
    "A4",
    "B4",
    "Bb4",
    "A4",
    "G4",
    "E5",
    "G5",
    "A5",
    "F5",
    "G5",
    "E5",
    "C5",
    "D5",
    "B4",
    "C5",
    "G4",
    "E4",
]

# Additional notes for the Super Mario theme song
super_mario_theme_part2 = [
    "B4", "C5", "D5", "E5", "C5", "A4", "A4", "A4",
    "G4", "F4", "G4", "A4", "B4", "A4", "G4", "E5",
    "G5", "A5", "F5", "G5", "E5", "C5", "D5", "B4",
    "C5", "G4", "E4", "A4", "B4", "Bb4", "A4", "G4",
    "E5", "G5", "A5", "F5", "G5", "E5", "C5", "D5",
    "B4", "C5", "G4", "E4", "E5", "E5", "E5", "C5",
    "E5", "G5", "G4", "C5", "G4", "E4", "A4", "B4",
    "Bb4", "A4", "G4", "E5", "G5", "A5", "F5", "G5",
    "E5", "C5", "D5", "B4", "C5", "G4", "E4"
]

# Combine the two parts of the Super Mario theme song

play_song(
    super_mario_theme_part2,
    note_duration,
    piano_frequencies,
)


# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["C4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# time.sleep(note_duration / 1000)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# time.sleep(note_duration / 1000)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["G4"], note_duration)
# play_pure_note(piano_frequencies["G4"], note_duration)
# time.sleep(note_duration / 1000)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["C4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["C4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["E4"], note_duration)
# play_pure_note(piano_frequencies["D4"], note_duration)
# play_pure_note(piano_frequencies["C4"], note_duration)


# c_wave = Sawtooth(piano_frequencies["C4"]).to_audio_segment(duration=note_duration)  # 1000 ms (1 second) duration
# e_wave = Sawtooth(piano_frequencies["E4"]).to_audio_segment(duration=note_duration)
# g_wave = Sawtooth(piano_frequencies["G4"]).to_audio_segment(duration=note_duration)

# # Combine the audio segments to create the C major chord
# chord = AudioSegment.from_mono_audiosegments(c_wave, e_wave, g_wave)

# # Play the chord
# play(chord)
