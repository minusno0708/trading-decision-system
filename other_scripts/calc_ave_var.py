import statistics
import math

data = [
    0.18963565657150586,
    0.37768455569454734,
    0.28586821413315244,
    0.8616101773937549,
    0.6195896787914345,
    0.2231726830760164,
    0.5089928057553957,
    0.5539568345323741,
    0.5161870503597122,
    0.8114740853843849
]

ave = statistics.mean(data)

pstdev = statistics.pstdev(data)

print(ave, pstdev)