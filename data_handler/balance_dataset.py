import matplotlib.pyplot as plt


def even_dataset(units: list, max_force: float = 2112, hist: bool = False):
    """
    Multiply high-force units and eliminate some low-force units
    :param hist: Set to True to plot histogram of original and new dataset
    :type hist: bool
    :param max_force: the highest force in dataset (default: 2112 for this dataset)
    :type max_force: int
    :param units: List of TrainUnit Objects
    :return: units list of TrainUnit Objects with increased dataset size
    """

    new_units = []
    low_forces = []

    for unit in units:

        if unit.force > max_force - 200:
            for i in range(0, 4):
                new_units.append(unit)
        elif unit.force > max_force - 400:
            for i in range(0, 3):
                new_units.append(unit)
        elif unit.force > max_force - 800:
            for i in range(0, 2):
                new_units.append(unit)
        elif unit.force > max_force - 1200:
            for i in range(0, 2):
                new_units.append(unit)
        elif unit.force > max_force - 1500:
            for i in range(0, 2):
                new_units.append(unit)
        elif unit.force > 150:
            new_units.append(unit)
        elif unit.force < 150:
            low_forces.append(unit)
        else:
            new_units.append(unit)

    for index, unit in enumerate(low_forces):
        if index % 5 == 0:
            new_units.append(unit)

    if hist:
        old_forces = [unit.force.squeeze() for unit in units]
        forces = [unit.force.squeeze() for unit in new_units]
        fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(10, 5))
        axs[0].hist(old_forces, bins=100)
        axs[1].hist(forces, bins=100)
        fig.text(0.5, 0.04, 'Force (N)', ha='center')
        plt.tight_layout(pad=3)
        fig.text(0.04, 0.5, 'Repetitions', va='center', rotation='vertical')
        plt.tight_layout(pad=5)
        axs[0].set_title('Original Distribution')
        axs[1].set_title('Evened out Distribution')
        plt.show()
    return new_units
