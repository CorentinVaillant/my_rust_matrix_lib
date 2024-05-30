# Stuff to do :

- make `self.get_inverse() -> Self`
- make `self.get_det() -> f64`
    - make `self.get_plus_decomposition() -> [Self;3]`
        - make `self.get_row_exange() -> Self`



```pseudo

function rowReduceEchelonForm(matrix):
    rows = number of rows in matrix
    cols = number of columns in matrix
    lead = 0

    for r = 0 to rows - 1:
        if lead >= cols:
            return matrix
        i = r
        while matrix[i][lead] == 0:
            i = i + 1
            if i == rows:
                i = r
                lead = lead + 1
                if lead >= cols:
                    return matrix
        swap rows i and r in matrix

        # Normalize the leading row
        leadValue = matrix[r][lead]
        for j = 0 to cols - 1:
            matrix[r][j] = matrix[r][j] / leadValue

        # Eliminate the column entries
        for i = 0 to rows - 1:
            if i != r:
                leadValue = matrix[i][lead]
                for j = 0 to cols - 1:
                    matrix[i][j] = matrix[i][j] - leadValue * matrix[r][j]
        
        lead = lead + 1

    return matrix

# Example usage:
# matrix = [[1, 2, 1, -1], [3, 8, 1, 4], [0, 4, 1, 0]]
# rref_matrix = rowReduceEchelonForm(matrix)

```