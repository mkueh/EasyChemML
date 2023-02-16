import numpy as np


class ReorderInplace:

    @staticmethod
    def put_at(index, axis=-1, slc=(slice(None),)):
        """Gets the numpy indexer for the given index based on the axis."""
        return (axis < 0) * (Ellipsis,) + axis * slc + (index,) + (-1 - axis) * slc

    @staticmethod
    def reorder_inplace(array, new_order, axis=0):
        """
        Reindex (reorder) the array along an axis.

        :param array: The array to reindex.
        :param new_order: A list with the new index order. Must be a valid permutation.
        :param axis: The axis to reindex.
        """
        if np.size(array, axis=axis) != len(new_order):
            raise ValueError(
                'The new order did not match indexed array along dimension %{0}; '
                'dimension is %{1} but corresponding boolean dimension is %{2}'.format(
                    axis, np.size(array, axis=axis), len(new_order)
                )
            )

        visited = set()
        for index, source in enumerate(new_order):
            if index not in visited and index != source:
                initial_values = np.take(array, index, axis=axis).copy()

                destination = index
                visited.add(destination)
                while source != index:
                    if source in visited:
                        raise IndexError(
                            'The new order is not unique; '
                            'duplicate found at position %{0} with value %{1}'.format(
                                destination, source
                            )
                        )

                    array[ReorderInplace.put_at(destination, axis=axis)] = array.take(source, axis=axis)

                    destination = source
                    source = new_order[destination]

                    visited.add(destination)
                array[ReorderInplace.put_at(destination, axis=axis)] = initial_values
