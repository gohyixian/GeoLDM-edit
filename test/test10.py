low = -4000
high = 4000
range_min = -1
range_max = 1

m = (range_max - range_min) / (high - low)
print(m)

c = range_min - (m * low)
print(c)

