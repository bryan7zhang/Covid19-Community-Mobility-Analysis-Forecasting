import matplotlib.pyplot as plt
from MLmodels import dataselector

data, title = dataselector.chooseDataset()

print(data)

numtoday_subset = data.loc[:, 'numtoday']
numtested_subset = data.loc[:, 'ratetested']
numtotal_subset = data.loc[:, 'numtotal']
subsetArr = ['retail_and_recreation',
             'grocery_and_pharmacy',
             'parks',
             'transit_stations',
             'workplaces',
             'residential']

# Num tested vs Num caught
plt.plot(numtested_subset, label="Tested")
plt.plot(numtotal_subset, label="Total Cases")
plt.suptitle('Tested vs Total Cases for {}'.format(title))
plt.legend()
plt.show()

# DAILY CASES
plt.plot(numtoday_subset, label="total cases today")
plt.suptitle('Total Cases per day for {}'.format(title))
plt.legend()
plt.show()

# DAILY CASES WITH COVID
fig, ax = plt.subplots()
ax.plot(numtoday_subset, color="red", label="Daily Cases")
ax.set_xlabel("day", fontsize=14)

ax2 = ax.twinx()
# ax2.plot(data.loc[:, subsetArr], color="blue")
ax2.plot(data.loc[:, subsetArr[0]], label="retail")
ax2.plot(data.loc[:, subsetArr[1]], label="grocery")
ax2.plot(data.loc[:, subsetArr[2]], label="parks")
ax2.plot(data.loc[:, subsetArr[3]], label='transit stations')
ax2.plot(data.loc[:, subsetArr[4]], label='workplaces')
ax2.plot(data.loc[:, subsetArr[5]], label='residential')
# plt.suptitle(title)
plt.legend()
# plt.xlim(200, 225)
plt.show()

# Locations
plt.plot(data.loc[:, subsetArr[0]], label="retail")
plt.plot(data.loc[:, subsetArr[1]], label="grocery")
plt.plot(data.loc[:, subsetArr[2]], label="parks")
plt.plot(data.loc[:, subsetArr[3]], label='transit stations')
plt.plot(data.loc[:, subsetArr[4]], label='workplaces')
plt.plot(data.loc[:, subsetArr[5]], label='residential')
plt.legend()
plt.suptitle('Mobility Trends for {}'.format(title))
plt.show()

for subset_title in subsetArr:
    subset = data.loc[:, subset_title]
    fig, ax = plt.subplots()
    ax.plot(numtoday_subset, color="red")
    ax.set_xlabel("day", fontsize=14)
    ax.set_ylabel("Num cases today", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(subset, color="blue")
    ax2.set_ylabel(subset_title, color="blue", fontsize=14)
    plt.suptitle(title)
    plt.show()

    # plt.bar(data.loc[:, 'day_of_week'], subset)
    # plt.suptitle("Day of the week mobility for {} ".format(subset_title))
    # plt.show()
