import json

filename = "out/solution-018d8b2c-b37b-782c-8cd5-cff77dba75b1.json"
contents = open(filename, 'r').read()

# Remove all instances of "Angles"
contents = contents.replace("Angles", "")

# Quote the words "elevation" and "direction"
contents = contents.replace("elevation", "\"elevation\"")
contents = contents.replace("direction", "\"direction\"")

# Parse as JSON
data = json.loads(contents)

# Truncate the array for readibility
data = data[:10]

# Transform the array of objects into an object of arrays
data = {k: [i[k] for i in data] for k in data[0]}

# Plot the data
import matplotlib.pyplot as plt
for k in data:
    plt.plot(data[k], label=k)
plt.legend()
plt.show()
