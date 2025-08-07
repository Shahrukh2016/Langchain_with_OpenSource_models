from langchain.text_splitter import RecursiveCharacterTextSplitter

text = '''
**Title: *Into the Abyss***

In the year 2178, humanity had reached the outer rim of the galaxy, where the mysterious black hole *Xerion-9* lurked like a sleeping giant. Dr. Alina Verma, a physicist aboard the deep-space vessel *Odyssey*, had one goal: to understand what lay beyond the event horizon. The ship approached cautiously, instruments trembling as gravity warped reality around them. Alina’s eyes widened—not in fear, but awe—as she witnessed stars stretch and bend like melting glass.

As *Odyssey* drifted closer, strange signals echoed from within the black hole—patterns that resembled music or code. Against protocol, Alina initiated the “Threshold Dive,” sending a probe into the swirling dark. Moments later, the ship was flooded with visions—memories not their own, languages never spoken, and images of civilizations long gone. Time lost all meaning. Some crew reported seeing their past and future simultaneously, while others heard whispers in their minds.

Then, silence. The ship was gone—at least from the eyes of the outside universe. But within the black hole, something remained. A spark, a seed of human consciousness suspended in timeless dark. Alina’s last log read: *“It’s not the end. It’s a doorway.”* And so, the story of *Odyssey* became legend—not of a crew lost, but of minds reborn inside the strangest cradle the cosmos ever made.

'''

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

# Perform the split
chunks = splitter.split_text(text=text)
print(len(chunks))
print(chunks)