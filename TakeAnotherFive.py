# Dante Danelian
# March 12th, 2022
# "Take Another Five" Solo
# CS109 Challenge Submission
# Dedicated to Haven Whitney and Ramya Ayyagari, who kept me company
# at Stanford Law School when I stayed up until 3AM to code this.
from music21 import note, stream, converter
from scipy import stats
import numpy as np

# CUSTOM PARAMETERS: Feel free to play around with these!

SOLO_LENGTH = 240
# The number of beats in the solo, including notes and rests.

LENGTH_P = 0.5
# p-value for note length Geometric distribution. For default value use 0.5.

REST_P = 0.9
# p-value for a Bernoulli which determines whether a note is a rest or not. A lower value means
# less notes and more rests. After all, jazz is about the notes you *don't* play.

SCALE = ['F#3', 'G#3', 'B-3', 'C4', 'C#4', 'D4', 'E-4', 'F4', 'F#4', 'G#4',
         'A4', 'B-4', 'B4', 'C5', 'C#5', 'D5', 'E-5', 'F5', 'F#5']
# The notes we use for the solo. Mainly taken from E-flat dorian, the scale used in "Take Five"


"""
Calls helper function createsolo() to get a list of notes, then writes this solo to a midi file.
"""
def main():
    s = createsolo(createdistribution('original-solo.mid'))
    s.write('midi', fp='new_solo6.midi')


"""
Uses the music21 library to initialize and populate a stream, which is a essentially a list of 
notes. For each note it adds to the stream, length is determined by helper function find_length(), 
and pitch is determined by dist parameter.
"""
def createsolo(dist):
    s = stream.Stream()
    count = 0
    pitch = 2
    while count < SOLO_LENGTH:
        length = find_length()
        if stats.bernoulli.rvs(REST_P):
            # sample from dist using values as probabilities
            pitch = np.random.choice(SCALE, 1, p=dist[pitch])
            new_note = note.Note(pitch[0], quarterLength=length)
            pitch = SCALE.index(pitch)
        else:   # Bernoulli has determined this note will be a rest instead.
            new_note = note.Rest(quarterLength=length)
        count += length
        s.append(new_note)

    return s


"""
Samples from a Geometric distribution using global variable LENGTH_P as a parameter. Returns a 
note length according to sample value. If X > 4, note length defaults to an eight note.
"""
def find_length():
    length = stats.geom.rvs(LENGTH_P)
    if length == 2:
        return 1
    elif length == 3:
        return 2
    elif length == 4:
        return 4
    else:
        return 0.5


"""
Takes in midi file of the solo we want to emulate. Parses through solo and counts how many times 
each note (i) is followed by each other note (j). Returns these counts as a normalized 
conditional probability distribution (stored in an np array), where rows are current notes, 
columns are potential following notes, and cells are the probability of a following note given 
a current note.
"""
def createdistribution(file):
    # Turn Take Five midi into a stream using music21.
    solo = converter.parse(file)

    pitches = {}
    last = note.Note('B-3')

    # Make dictionary where keys are notes in our SCALE.
    for i in SCALE:
        pitches[i] = {}
        for j in SCALE:
            pitches[i][j] = 0

    # For each note (i), count how many times it is followed by each other note (j).
    for curr_note in solo[note.Note]:
        pitches[str(last.pitch)][str(curr_note.pitch)] += 1
        last = curr_note

    # Transform dictionary into numpy array where vertical axis is current notes and horizontal
    # axis is potential next notes.
    distribution = np.ndarray((len(SCALE), len(SCALE)))
    for i in range(len(SCALE)):
        for j in range(len(SCALE)):
            distribution[i, j] = pitches[SCALE[i]][SCALE[j]]

    # Normalize distribution and return it.
    for i in range(len(SCALE)):
        distribution[i] /= np.sum(distribution[i])

    return distribution

if __name__ == '__main__':
    main()
