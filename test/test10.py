low = -4000
high = 4000
range_min = 0
range_max = 1

m = (range_max - range_min) / (high - low)
print(m)

c = abs(m*low)
print(c)

