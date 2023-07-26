a = scipy.sparse.dok_array(
    np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                2,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4],
            [0, 0, 0, 3, 0, 0],
        ], dtype=np.complex128
    )
)

# Find rows and columns to remove
row_sums = np.sum(a, axis=1)
col_sums = np.sum(a, axis=0)
rows_to_keep = ~np.isclose(row_sums, 0.0)
cols_to_keep = ~np.isclose(col_sums, 0.0)

# Apply boolean mask to CSR matrix
filtered_matrix = a[rows_to_keep][:, cols_to_keep]

print(filtered_matrix)