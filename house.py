import matplotlib.pyplot as plt

rents = [10000, 13000, 13000, 15000, 15000, 13000]
months = [1, 2, 3, 4, 5, 6]

plt.plot(months, rents)
plt.xlabel('Months')
plt.ylabel('Rennt (Rs.)')
plt.title('Rent of the house over 6 months')
plt.show()
